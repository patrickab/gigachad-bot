/**
 * The `/sessions/:id/…` routes of the Supervisor and the Catalog: one path parser and one method table.
 *
 * - Consumes: a request's URL and method
 * - Provides: `sessionRoute`
 */

/** The method of each route by action: `""` is `/sessions/:id`, `turns` is `/sessions/:id/turns/:entryId/delete`. */
const METHODS = { "": "DELETE", close: "POST", turns: "POST", archive: "POST", restore: "POST", pin: "PUT", entries: "GET", matches: "GET" } as const;
const SESSION_PATH = /^\/sessions\/([^/]+)(?:\/(close|archive|restore|pin|entries|matches)|\/turns\/([^/]+)\/delete)?$/;

export type SessionRoute = { sessionId: string; action: keyof typeof METHODS; entryId?: string };

/** The route of a `/sessions/:id/…` path, undefined for another path; 405 for another method, 400 for an id that is not valid percent-encoding. */
export function sessionRoute(url: URL, method: string): SessionRoute | Response | undefined {
  const match = SESSION_PATH.exec(url.pathname);
  if (!match) return undefined;
  const [, id, named, entry] = match;
  const action = (named ?? (entry === undefined ? "" : "turns")) as SessionRoute["action"];
  if (METHODS[action] !== method) return Response.json({ error: "method_not_allowed" }, { status: 405 });
  try {
    return { sessionId: decodeURIComponent(id!), action, ...(entry === undefined ? {} : { entryId: decodeURIComponent(entry) }) };
  } catch {
    return Response.json({ error: "invalid_session_id" }, { status: 400 });
  }
}
