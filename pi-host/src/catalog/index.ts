/**
 * Catalog HTTP routes: session listing, archive/restore, pin, private state and `@` file completion.
 *
 * - Consumes: Pi session files (see `paths.ts`), `LiveSessions` from the supervisor, git in the requested cwd
 * - Provides: the Catalog rows of the pi-host HTTP contract, plus `sessionInfo`, `isPinned` and
 *   `copyPrivateState` for the supervisor
 */
import { stat } from "node:fs/promises";
import { isAbsolute, resolve } from "node:path";
import type { LiveSessions } from "../supervisor/index.ts";
import { listFiles, rankFiles } from "./files.ts";
import { findSession, listSessions, moveSession, sessionInfo } from "./sessions.ts";
import { isPlainObject, readState, updateState } from "./state.ts";

export { sessionInfo, type LiveStatus, type SessionInfo } from "./sessions.ts";
export { copyPrivateState, isPinned } from "./state.ts";

const DEFAULT_FILE_LIMIT = 50;
const MAX_FILE_LIMIT = 500;
const SESSION_ACTION = /^\/sessions\/([^/]+)\/(archive|restore|pin|state)$/;

/** JSON error response `{error}`. */
function fail(status: number, error: string): Response {
  return Response.json({ error }, { status });
}

/** Resolved `cwd` when it is an absolute path to an existing directory. */
async function validCwd(raw: string | null): Promise<string | undefined> {
  if (!raw || !isAbsolute(raw)) return undefined;
  const cwd = resolve(raw);
  try {
    return (await stat(cwd)).isDirectory() ? cwd : undefined;
  } catch {
    return undefined;
  }
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
async function handleList(url: URL, live: LiveSessions): Promise<Response> {
  const cwd = await validCwd(url.searchParams.get("cwd"));
  if (!cwd) return fail(400, "invalid_cwd");
  const archived = url.searchParams.get("archived") === "1";
  return Response.json({ sessions: await listSessions(cwd, archived, live) });
}

/** `GET /files?cwd=&q=&limit=`. */
async function handleFiles(url: URL): Promise<Response> {
  const cwd = await validCwd(url.searchParams.get("cwd"));
  if (!cwd) return fail(400, "invalid_cwd");
  const requested = Number(url.searchParams.get("limit") ?? DEFAULT_FILE_LIMIT);
  const limit = Number.isInteger(requested) ? Math.min(Math.max(requested, 1), MAX_FILE_LIMIT) : DEFAULT_FILE_LIMIT;
  const files = rankFiles(await listFiles(cwd), url.searchParams.get("q") ?? "", limit);
  return Response.json({ files });
}

/** `POST /sessions/:id/archive` and `/restore`, 409 while live or when the target exists. */
async function handleMove(sessionPath: string, toArchive: boolean, live: LiveSessions): Promise<Response> {
  const moved = await moveSession(sessionPath, toArchive, live);
  if (!moved.ok) {
    if (moved.error === "not_in_root") return fail(409, toArchive ? "already_archived" : "not_archived");
    return fail(moved.error === "not_found" ? 404 : 409, moved.error);
  }
  const session = await sessionInfo(moved.path, live);
  return session ? Response.json({ session }) : fail(404, "not_found");
}

/** `PUT /sessions/:id/pin` with `{pinned}`. */
async function handlePin(request: Request, sessionPath: string, live: LiveSessions): Promise<Response> {
  const body = await jsonBody(request);
  if (!isPlainObject(body) || typeof body.pinned !== "boolean") return fail(400, "invalid_body");
  const pinned = body.pinned;
  await updateState(sessionPath, (state) => ({ next: { ...state, pinned }, result: undefined }));
  const session = await sessionInfo(sessionPath, live);
  return session ? Response.json({ session }) : fail(404, "not_found");
}

/** `PUT /sessions/:id/state` with `{revision, data}`, 409 `{revision, data}` on a stale revision. */
async function handlePutState(request: Request, sessionPath: string): Promise<Response> {
  const body = await jsonBody(request);
  if (!isPlainObject(body) || !Number.isInteger(body.revision) || !isPlainObject(body.data)) {
    return fail(400, "invalid_body");
  }
  const revision = body.revision as number;
  const data = body.data;
  return updateState(sessionPath, (state) => {
    if (state.revision !== revision) {
      return { result: Response.json({ revision: state.revision, data: state.data }, { status: 409 }) };
    }
    const next = { ...state, revision: state.revision + 1, data };
    return { next, result: Response.json({ revision: next.revision }) };
  });
}

/** Route a Catalog request, undefined when the route is not the Catalog's. */
export async function handleCatalog(request: Request, url: URL, live: LiveSessions): Promise<Response | undefined> {
  if (url.pathname === "/sessions" && request.method === "GET") return handleList(url, live);
  if (url.pathname === "/files" && request.method === "GET") return handleFiles(url);
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
  const sessionPath = await findSession(sessionId);
  if (!sessionPath) return fail(404, "not_found");
  if (action === "archive" || action === "restore") return handleMove(sessionPath, action === "archive", live);
  if (action === "pin") return handlePin(request, sessionPath, live);
  if (request.method === "PUT") return handlePutState(request, sessionPath);
  const { revision, data } = await readState(sessionPath);
  return Response.json({ revision, data });
}
