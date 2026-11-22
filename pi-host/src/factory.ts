/**
 * Factory routes: the factory plugin's core, called through HTTP. pi-host adds no factory logic of its own.
 *
 * - Consumes: `<agent dir>/extensions/factory/core.ts` (the plugin's only client module, plain JSON in and out),
 *   imported on first use and again when its mtime changes, so a pi-host started without the plugin finds it once
 *   installed and a plugin update reaches it without a restart (the core's own imports stay cached until one)
 * - Provides: `handleFactory`, the rows `POST /factory/call` and `GET /factory/actions` of the pi-host HTTP contract
 *
 * A runner an action starts inherits pi-host's environment, as a task started from a terminal inherits the shell's
 * (user decision: a factory task has everything Pi has).
 */
import { stat } from "node:fs/promises";
import { join } from "node:path";
import { getAgentDir } from "@earendil-works/pi-coding-agent";

/** Where the factory plugin comes from. It is a Pi user extension, not part of this repo. */
export const FACTORY_PLUGIN_SOURCE = "https://github.com/patrickab/.os/tree/master/chezmoi/dot_pi/agent/extensions/factory";

/** HTTP status of each `FactoryError` code. */
const STATUS_OF_CODE: Record<string, number> = { invalid: 400, not_found: 404, busy: 409, refused: 409, failed: 500 };

/** The method each factory route takes. */
const ROUTE_METHODS: Record<string, string> = { "/factory/call": "POST", "/factory/actions": "GET" };

/** The part of `core.ts` pi-host uses. `env` is the environment of a runner the action starts. */
type FactoryCore = {
  ACTIONS: Record<string, unknown>;
  callAction: (name: string, params: unknown, options?: { env?: Record<string, string> }) => Promise<unknown>;
  FactoryError: new (...args: never[]) => Error & { code: string; detail?: Record<string, unknown> };
};

function factoryCorePath(): string {
  return join(getAgentDir(), "extensions", "factory", "core.ts");
}

/** JSON error response `{error, message, detail?}`. */
function fail(status: number, error: string, message: string, detail?: Record<string, unknown>): Response {
  return Response.json(detail === undefined ? { error, message } : { error, message, detail }, { status });
}

/**
 * The plugin's core, or the 503 `factory_plugin_missing` answer naming its source when it is not installed. The
 * mtime in the specifier makes Bun import a changed `core.ts` anew and reuse the module while it is unchanged.
 */
async function loadCore(): Promise<FactoryCore | Response> {
  const path = factoryCorePath();
  try {
    const { mtimeMs } = await stat(path);
    return (await import(`${path}?mtime=${mtimeMs}`)) as FactoryCore;
  } catch {
    return fail(503, "factory_plugin_missing", `The factory plugin is not installed (${path}). Install it from ${FACTORY_PLUGIN_SOURCE}`);
  }
}

/**
 * `POST /factory/call {action, params}`: 200 `{result}`; a `FactoryError` answers `{error: code, message, detail?}`
 * with 400 invalid, 404 not_found, 409 busy|refused, 500 failed.
 */
async function handleCall(request: Request, core: FactoryCore): Promise<Response> {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return fail(400, "invalid", "The body is not JSON.");
  }
  if (typeof body !== "object" || body === null || typeof (body as { action?: unknown }).action !== "string") {
    return fail(400, "invalid", "The body needs an action name.");
  }
  const { action, params } = body as { action: string; params?: unknown };
  try {
    return Response.json({ result: (await core.callAction(action, params)) ?? null });
  } catch (error) {
    if (error instanceof core.FactoryError) return fail(STATUS_OF_CODE[error.code] ?? 500, error.code, error.message, error.detail);
    return fail(500, "failed", error instanceof Error ? error.message : String(error));
  }
}

/** Route a factory request, undefined when the route is not a factory one. */
export async function handleFactory(request: Request, url: URL): Promise<Response | undefined> {
  const method = ROUTE_METHODS[url.pathname];
  if (method === undefined) return undefined;
  if (request.method !== method) return Response.json({ error: "method_not_allowed" }, { status: 405 });
  const core = await loadCore();
  if (core instanceof Response) return core;
  return url.pathname === "/factory/call" ? handleCall(request, core) : Response.json({ actions: core.ACTIONS });
}
