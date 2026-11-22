/**
 * Catalog HTTP routes: session listing, search, archive/restore, pin, private state, persisted entries, `@` file completion and the factory draft.
 *
 * - Consumes: a `Host` for every session file, sidecar and file listing, `LiveSessions` from the supervisor
 * - Provides: the Catalog rows of the pi-host HTTP contract, plus `sessionInfo`, `isPinned`, `copyPrivateState`,
 *   `profileOf` and `markProfile` for the supervisor
 */
import { isAbsolute, join, resolve } from "node:path";
import { getAgentDir } from "@earendil-works/pi-coding-agent";
import { PROFILES, type Host, type Profile } from "../host/index.ts";
import type { LiveSessions } from "../supervisor/index.ts";
import { sessionEntries } from "./entries.ts";
import { rankFiles } from "./files.ts";
import { listSessions, sessionInfo, type SessionInfo } from "./sessions.ts";
import { isPlainObject, updateState } from "./state.ts";

export { listSessions, sessionInfo, type LiveStatus, type SessionInfo } from "./sessions.ts";
export { copyPrivateState, isPinned, markProfile, profileOf } from "./state.ts";

const DEFAULT_FILE_LIMIT = 50;
const MAX_FILE_LIMIT = 500;
const DEFAULT_SEARCH_LIMIT = 20;
const MAX_SEARCH_LIMIT = 100;
/** The methods each `/sessions/:id/<action>` route takes. */
const ACTION_METHODS = {
  archive: ["POST"],
  restore: ["POST"],
  pin: ["PUT"],
  state: ["GET", "PUT"],
  entries: ["GET"],
} as const satisfies Record<string, readonly string[]>;
type SessionAction = keyof typeof ACTION_METHODS;
const SESSION_ACTION = new RegExp(`^/sessions/([^/]+)/(${Object.keys(ACTION_METHODS).join("|")})$`);

/** One `GET /search` hit (M24). */
export type SearchHit = { session: SessionInfo; snippet: string; entryId: string };

/** JSON error response `{error}`. */
function fail(status: number, error: string): Response {
  return Response.json({ error }, { status });
}

/** Resolved `cwd` when it is an absolute path to an existing directory on the host. */
async function validCwd(host: Host, raw: string | null): Promise<string | undefined> {
  if (!raw || !isAbsolute(raw)) return undefined;
  const cwd = resolve(raw);
  return (await host.isDirectory(cwd)) ? cwd : undefined;
}

/** Request JSON body, undefined when it is not valid JSON. */
async function jsonBody(request: Request): Promise<unknown> {
  try {
    return await request.json();
  } catch {
    return undefined;
  }
}

/** Integer `limit` query parameter clamped to `1..max`, `fallback` when absent or not an integer. */
function limitParam(url: URL, fallback: number, max: number): number {
  const requested = Number(url.searchParams.get("limit") ?? fallback);
  return Number.isInteger(requested) ? Math.min(Math.max(requested, 1), max) : fallback;
}

/**
 * `GET /sessions?cwd=&archived=0|1&profile=&factory=0|1`, every cwd's sessions without `cwd`, every profile's without
 * `profile`. `factory=1` adds the factory root's sessions (the factory's own listing); without it none of them.
 */
async function handleList(url: URL, live: LiveSessions, host: Host): Promise<Response> {
  const rawCwd = url.searchParams.get("cwd");
  const cwd = rawCwd === null ? null : await validCwd(host, rawCwd);
  if (cwd === undefined) return fail(400, "invalid_cwd");
  const profile = url.searchParams.get("profile") ?? undefined;
  if (profile !== undefined && !PROFILES.includes(profile as Profile)) return fail(400, "invalid_profile");
  const archived = url.searchParams.get("archived") === "1";
  const factory = url.searchParams.get("factory") === "1";
  return Response.json({ sessions: await listSessions(host, cwd, archived, live, profile as Profile | undefined, factory) });
}

/**
 * `GET /files?cwd=&q=&limit=`: ranked against `q`, or without `q` the cached list unranked (K6: the UI ranks it),
 * cut to `limit` only when one is given.
 */
async function handleFiles(url: URL, host: Host): Promise<Response> {
  const cwd = await validCwd(host, url.searchParams.get("cwd"));
  if (!cwd) return fail(400, "invalid_cwd");
  const query = url.searchParams.get("q") ?? "";
  const candidates = await host.listFiles(cwd);
  if (!query.trim()) {
    const listed = url.searchParams.has("limit") ? candidates.slice(0, limitParam(url, DEFAULT_FILE_LIMIT, MAX_FILE_LIMIT)) : candidates;
    return Response.json({ files: listed.map((path) => ({ path, score: 0 })) });
  }
  return Response.json({ files: rankFiles(candidates, query, limitParam(url, DEFAULT_FILE_LIMIT, MAX_FILE_LIMIT)) });
}

/** Where the factory plugin comes from. It is a Pi user extension, not part of this repo. */
export const FACTORY_PLUGIN_SOURCE = "https://github.com/patrickab/.os/tree/master/chezmoi/dot_pi/agent/extensions/factory";

/** The factory plugin's generator module in Pi's user extensions. Loaded on first use, so a pi-host started without the plugin finds it once installed. */
function factoryPluginPath(): string {
  return join(getAgentDir(), "extensions", "factory", "draft.ts");
}

/**
 * `GET /factory/draft?cwd=`: the factory plugin's draft `factory.yaml` for the checkout at `cwd`, nothing is written.
 * 503 `factory_plugin_missing` with a `message` naming the plugin's source when it is not installed.
 */
async function handleFactoryDraft(url: URL, host: Host): Promise<Response> {
  const cwd = await validCwd(host, url.searchParams.get("cwd"));
  if (!cwd) return fail(400, "invalid_cwd");
  const path = factoryPluginPath();
  let plugin: { initDraft: (root: string) => { text: string; source: string } };
  try {
    plugin = await import(path);
  } catch {
    const message = `The factory plugin is not installed (${path}). Install it from ${FACTORY_PLUGIN_SOURCE}`;
    return Response.json({ error: "factory_plugin_missing", message }, { status: 503 });
  }
  const { text, source } = plugin.initDraft(cwd);
  return Response.json({ draft: text, source });
}

/** `GET /search?q=&limit=20` (M24): live sessions whose transcript matches `q`, best first, factory tasks left out. */
async function handleSearch(url: URL, live: LiveSessions, host: Host): Promise<Response> {
  const query = url.searchParams.get("q") ?? "";
  if (!query.trim()) return Response.json({ hits: [] });
  const matches = await host.searchSessions(query, limitParam(url, DEFAULT_SEARCH_LIMIT, MAX_SEARCH_LIMIT));
  const hits = await Promise.all(
    matches.map(async ({ sessionPath, snippet, entryId }): Promise<SearchHit | undefined> => {
      const session = await sessionInfo(host, sessionPath, live);
      return session && { session, snippet, entryId };
    }),
  );
  return Response.json({ hits: hits.filter((hit) => hit !== undefined) });
}

/**
 * `POST /sessions/:id/archive` and `/restore` (C7): a detached idle child is closed first. 409 `live` while a child runs,
 * has a client or asks a question, or a spawning child holds the file; 409 too when the target exists. 409 `factory`
 * for a factory-root session, which is never archived.
 */
async function handleMove(sessionPath: string, toArchive: boolean, live: LiveSessions, host: Host): Promise<Response> {
  const meta = await host.sessionMeta(sessionPath);
  if (!meta) return fail(404, "not_found");
  if (meta.archived === toArchive) return fail(409, toArchive ? "already_archived" : "not_archived");
  // Either path form may key the child: a new session's as Pi named it, a resumed one's real path
  for (const path of new Set([meta.sessionPath, meta.realPath])) {
    if (!(await live.closeIfIdle(path))) return fail(409, "live");
  }
  const moved = await host.moveSession(sessionPath, toArchive);
  if (!moved.ok) {
    if (moved.error === "not_in_root") return fail(409, toArchive ? "already_archived" : "not_archived");
    if (moved.error === "not_found") return fail(404, "not_found");
    return fail(409, moved.error === "locked" ? "live" : moved.error);
  }
  const session = await sessionInfo(host, moved.path, live);
  return session ? Response.json({ session }) : fail(404, "not_found");
}

/** `PUT /sessions/:id/pin` with `{pinned}`. */
async function handlePin(request: Request, sessionPath: string, live: LiveSessions, host: Host): Promise<Response> {
  const body = await jsonBody(request);
  if (!isPlainObject(body) || typeof body.pinned !== "boolean") return fail(400, "invalid_body");
  const pinned = body.pinned;
  await updateState(host, sessionPath, (state) => ({ next: { ...state, pinned }, result: undefined }));
  const session = await sessionInfo(host, sessionPath, live);
  return session ? Response.json({ session }) : fail(404, "not_found");
}

/** `PUT /sessions/:id/state` with `{revision, data}`, 409 `{revision, data}` on a stale revision. */
async function handlePutState(request: Request, sessionPath: string, host: Host): Promise<Response> {
  const body = await jsonBody(request);
  if (!isPlainObject(body) || !Number.isInteger(body.revision) || !isPlainObject(body.data)) {
    return fail(400, "invalid_body");
  }
  const revision = body.revision as number;
  const data = body.data;
  return updateState(host, sessionPath, (state) => {
    if (state.revision !== revision) {
      return { result: Response.json({ revision: state.revision, data: state.data }, { status: 409 }) };
    }
    const next = { ...state, revision: state.revision + 1, data };
    return { next, result: Response.json({ revision: next.revision }) };
  });
}

/**
 * `GET /sessions/:id/entries`: the session file's entries. `?turns=N` answers a preview of the active branch's last N
 * turns. `?customType=<prefix>&since=<entryId>` answers only the `custom` entries whose `customType` starts with the
 * prefix and only those appended after the entry, 400 `invalid_since` when the file has no such entry. `&assistant=1`
 * adds the assistant answers among them, cut to text, model and usage (`assistantAnswer`). `turns` takes
 * neither of the two.
 */
async function handleEntries(url: URL, sessionPath: string, host: Host): Promise<Response> {
  const raw = url.searchParams.get("turns");
  const turns = raw === null ? undefined : Number(raw);
  if (turns !== undefined && !(Number.isInteger(turns) && turns > 0)) return fail(400, "invalid_turns");
  const customType = url.searchParams.get("customType") ?? undefined;
  const since = url.searchParams.get("since") ?? undefined;
  const assistant = url.searchParams.get("assistant") === "1";
  if (turns !== undefined && (customType !== undefined || since !== undefined || assistant)) return fail(400, "invalid_query");
  const answer = await sessionEntries(host, sessionPath, { turns, customType, since, assistant });
  return answer ? Response.json(answer) : fail(400, "invalid_since");
}

/** Route a Catalog request, undefined when the route is not the Catalog's. */
export async function handleCatalog(request: Request, url: URL, live: LiveSessions, host: Host): Promise<Response | undefined> {
  if (url.pathname === "/sessions" && request.method === "GET") return handleList(url, live, host);
  if (url.pathname === "/files" && request.method === "GET") return handleFiles(url, host);
  if (url.pathname === "/factory/draft" && request.method === "GET") return handleFactoryDraft(url, host);
  if (url.pathname === "/search" && request.method === "GET") return handleSearch(url, live, host);
  const match = SESSION_ACTION.exec(url.pathname);
  if (!match) return undefined;
  const [, rawId, rawAction] = match;
  const action = rawAction as SessionAction;
  if (!(ACTION_METHODS[action] as readonly string[]).includes(request.method)) return fail(405, "method_not_allowed");
  let sessionId: string;
  try {
    sessionId = decodeURIComponent(rawId!);
  } catch {
    return fail(400, "invalid_session_id");
  }
  const sessionPath = await host.findSession(sessionId);
  if (!sessionPath) return fail(404, "not_found");
  if (action === "archive" || action === "restore") return handleMove(sessionPath, action === "archive", live, host);
  if (action === "pin") return handlePin(request, sessionPath, live, host);
  if (action === "entries") return handleEntries(url, sessionPath, host);
  if (request.method === "PUT") return handlePutState(request, sessionPath, host);
  const { revision, data } = await host.readState(sessionPath);
  return Response.json({ revision, data });
}
