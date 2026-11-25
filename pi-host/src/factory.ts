/**
 * Factory routes: the agent-factory package's core, called through HTTP. pi-host adds no factory logic of its own.
 *
 * - Consumes: the package's `factory` CLI (on PATH, or env `AGENT_FACTORY_CLI`), asked once for the core's path with
 *   `factory --core-path`, and that `core.ts` (the package's only client module, plain JSON in and out), imported on
 *   first use and again when its mtime changes, so a pi-host started without the package finds it once installed and
 *   a package update reaches it without a restart (the core's own imports stay cached until one)
 * - Provides: `handleFactory`, the rows `POST /factory/call` and `GET /factory/actions` of the pi-host HTTP contract
 *
 * A runner an action starts inherits pi-host's environment, as a task started from a terminal inherits the shell's
 * (user decision: a factory task has everything Pi has).
 */
import { stat } from "node:fs/promises";
import { isAbsolute } from "node:path";

/** Where the factory comes from. It is a package of its own, not part of this repo. */
export const FACTORY_PLUGIN_SOURCE = "https://github.com/patrickab/agent-factory";

/** A `factory --core-path` that hangs must not hold a request forever. */
const CORE_PATH_TIMEOUT_MS = 10_000;

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

/** JSON error response `{error, message, detail?}`. */
function fail(status: number, error: string, message: string, detail?: Record<string, unknown>): Response {
  return Response.json(detail === undefined ? { error, message } : { error, message, detail }, { status });
}

/** The core's path the CLI named, kept per CLI command: the CLI runs once, not per request. */
let cachedCorePath: { cli: string; path: string } | undefined;

/** The core's path from `<cli> --core-path`; throws with the reason when the CLI is missing, fails or names none. */
async function factoryCorePath(cli: string): Promise<string> {
  if (cachedCorePath?.cli === cli) return cachedCorePath.path;
  const child = Bun.spawn([cli, "--core-path"], { stdin: "ignore", stdout: "pipe", stderr: "pipe", timeout: CORE_PATH_TIMEOUT_MS });
  const [stdout, stderr, exitCode] = await Promise.all([new Response(child.stdout).text(), new Response(child.stderr).text(), child.exited]);
  if (exitCode !== 0) throw new Error(`\`${cli} --core-path\` exited with ${exitCode}${stderr.trim() ? `: ${stderr.trim()}` : ""}`);
  const path = stdout.trim();
  if (!isAbsolute(path)) throw new Error(`\`${cli} --core-path\` printed no absolute path`);
  cachedCorePath = { cli, path };
  return path;
}

/**
 * The package's core, or the 503 `factory_plugin_missing` answer naming its source when it is not installed. Only a
 * path that imports is kept, so installing or moving the package later needs no restart. The mtime in the specifier
 * makes Bun import a changed `core.ts` anew and reuse the module while it is unchanged.
 */
async function loadCore(): Promise<FactoryCore | Response> {
  const cli = process.env.AGENT_FACTORY_CLI || "factory";
  try {
    const path = await factoryCorePath(cli);
    try {
      const { mtimeMs } = await stat(path);
      return (await import(`${path}?mtime=${mtimeMs}`)) as FactoryCore;
    } catch (error) {
      cachedCorePath = undefined;
      throw new Error(`${path}: ${error instanceof Error ? error.message : String(error)}`);
    }
  } catch (error) {
    const reason = error instanceof Error ? error.message : String(error);
    return fail(503, "factory_plugin_missing", `The factory is not installed (${reason}). Install it from ${FACTORY_PLUGIN_SOURCE}`);
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
