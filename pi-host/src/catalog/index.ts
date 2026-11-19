/**
 * Catalog HTTP routes: session listing, archive/restore, pin, private state and `@` file completion.
 *
 * - Consumes: a `Host` for every session file, sidecar and file listing, `LiveSessions` from the supervisor
 * - Provides: the Catalog rows of the pi-host HTTP contract, plus `sessionInfo`, `isPinned` and
 *   `copyPrivateState` for the supervisor
 */
import { isAbsolute, resolve } from "node:path";
import type { Host } from "../host/index.ts";
import type { LiveSessions } from "../supervisor/index.ts";
import { rankFiles } from "./files.ts";
import { listSessions, liveStatusOf, sessionInfo } from "./sessions.ts";
import { isPlainObject, updateState } from "./state.ts";

export { listSessions, sessionInfo, type LiveStatus, type SessionInfo } from "./sessions.ts";
export { copyPrivateState, isPinned } from "./state.ts";

const DEFAULT_FILE_LIMIT = 50;
const MAX_FILE_LIMIT = 500;
const SESSION_ACTION = /^\/sessions\/([^/]+)\/(archive|restore|pin|state)$/;

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

/** `GET /sessions?cwd=&archived=0|1`. */
async function handleList(url: URL, live: LiveSessions, host: Host): Promise<Response> {
  const cwd = await validCwd(host, url.searchParams.get("cwd"));
  if (!cwd) return fail(400, "invalid_cwd");
  const archived = url.searchParams.get("archived") === "1";
  return Response.json({ sessions: await listSessions(host, cwd, archived, live) });
}

/** `GET /files?cwd=&q=&limit=`. */
async function handleFiles(url: URL, host: Host): Promise<Response> {
  const cwd = await validCwd(host, url.searchParams.get("cwd"));
  if (!cwd) return fail(400, "invalid_cwd");
  const requested = Number(url.searchParams.get("limit") ?? DEFAULT_FILE_LIMIT);
  const limit = Number.isInteger(requested) ? Math.min(Math.max(requested, 1), MAX_FILE_LIMIT) : DEFAULT_FILE_LIMIT;
  const files = rankFiles(await host.listFiles(cwd), url.searchParams.get("q") ?? "", limit);
  return Response.json({ files });
}

/** `POST /sessions/:id/archive` and `/restore`: 409 while live (C7), when held by a spawning child or the target exists. */
async function handleMove(sessionPath: string, toArchive: boolean, live: LiveSessions, host: Host): Promise<Response> {
  const meta = await host.sessionMeta(sessionPath);
  if (!meta) return fail(404, "not_found");
  if (meta.archived === toArchive) return fail(409, toArchive ? "already_archived" : "not_archived");
  if (liveStatusOf(meta, live)) return fail(409, "live");
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

/** Route a Catalog request, undefined when the route is not the Catalog's. */
export async function handleCatalog(request: Request, url: URL, live: LiveSessions, host: Host): Promise<Response | undefined> {
  if (url.pathname === "/sessions" && request.method === "GET") return handleList(url, live, host);
  if (url.pathname === "/files" && request.method === "GET") return handleFiles(url, host);
  const match = SESSION_ACTION.exec(url.pathname);
  if (!match) return undefined;
  const [, rawId, action] = match;
  const allowed = action === "archive" || action === "restore" ? "POST" : action === "pin" ? "PUT" : undefined;
  if (allowed ? request.method !== allowed : request.method !== "GET" && request.method !== "PUT") {
    return fail(405, "method_not_allowed");
  }
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
  if (request.method === "PUT") return handlePutState(request, sessionPath, host);
  const { revision, data } = await host.readState(sessionPath);
  return Response.json({ revision, data });
}
