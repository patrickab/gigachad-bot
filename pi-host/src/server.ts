/**
 * pi-host: Pi session supervisor, completion gateway and usage for gigachad-bot.
 *
 * - Consumes: the pinned `pi` binary, ~/.pi/agent (auth, sessions, extensions), `PI_HOST_BIND_HOST`, `PI_HOST_PORT`,
 *   the full and gateway token files (`auth.ts`)
 * - Provides: the HTTP + WS contract in .docs/pi-migration-contracts.md, loopback only, bearer token (C1: the gateway
 *   token opens `/v1/*` only)
 *
 * One `Host` (M23, `LocalHost` for this machine) serves the supervisor and the catalog.
 */
import type { Server } from "bun";
import { authorize, loadToken, type Tokens } from "./auth.ts";
import { handleCatalog } from "./catalog/index.ts";
import { handleGateway } from "./gateway/index.ts";
import { stopOmp } from "./gateway/omp.ts";
import { LocalHost } from "./host/local/index.ts";
import { loadConfig } from "./supervisor/config.ts";
import { createSupervisor, type SocketData, type Supervisor } from "./supervisor/index.ts";

export type ServerOptions = { hostname: string; port: number; tokens: Tokens; supervisor: Supervisor };

/** A request answered slower than this is logged, so a stall names its route. */
const SLOW_REQUEST_MS = 5_000;
/** How often the event-loop watch ticks, and the lag past a tick that it logs. */
const LOOP_TICK_MS = 500;
const LOOP_STALL_MS = 1_000;

/**
 * Logs every time the event loop was blocked for longer than `LOOP_STALL_MS`: every request waits on it, so a
 * synchronous job that stalls it shows up here and not as a slow route.
 */
export function watchEventLoop(log: (message: string) => void = console.warn, clock: () => number = () => performance.now()): () => void {
  let expected = clock() + LOOP_TICK_MS;
  const timer = setInterval(() => {
    const now = clock();
    const lag = now - expected;
    if (lag > LOOP_STALL_MS) log(`pi-host: event loop blocked for ${(lag / 1000).toFixed(1)} s`);
    expected = now + LOOP_TICK_MS;
  }, LOOP_TICK_MS);
  timer.unref();
  return () => clearInterval(timer);
}

/** Serve every route behind the Origin and bearer check (C3, C1). */
export function startServer({ hostname, port, tokens, supervisor }: ServerOptions): Server<SocketData> {
  return Bun.serve<SocketData>({
    hostname,
    port,
    idleTimeout: 0,
    async fetch(request, server) {
      const url = new URL(request.url);
      const denied = authorize(request, url.pathname, tokens);
      if (denied) return denied;
      if (url.pathname === "/ws") {
        return server.upgrade(request, { data: supervisor.socketData() })
          ? undefined
          : new Response("upgrade failed", { status: 400 });
      }
      if (url.pathname === "/health") return Response.json(await supervisor.health());
      const started = performance.now();
      const response =
        (await handleGateway(request, url)) ??
        (await supervisor.handle(request, url)) ??
        (await handleCatalog(request, url, supervisor, supervisor.host)) ??
        new Response("not found", { status: 404 });
      const elapsed = performance.now() - started;
      if (elapsed > SLOW_REQUEST_MS) console.warn(`pi-host: ${request.method} ${url.pathname} took ${(elapsed / 1000).toFixed(1)} s (${response.status})`);
      return response;
    },
    websocket: supervisor.websocket,
  });
}

if (import.meta.main) {
  const config = loadConfig();
  const tokens = { full: await loadToken(), gateway: await loadToken(config.gatewayTokenPath) };
  let supervisor: Supervisor;
  try {
    supervisor = await createSupervisor({ host: new LocalHost(config) });
  } catch (error) {
    console.error(`pi-host: ${error instanceof Error ? error.message : String(error)}`);
    process.exit(1);
  }
  const server = startServer({
    hostname: process.env.PI_HOST_BIND_HOST ?? "127.0.0.1",
    port: config.gatewayPort,
    tokens,
    supervisor,
  });
  console.log(`pi-host listening on http://${server.hostname}:${server.port}`);
  watchEventLoop();
  for (const signal of ["SIGINT", "SIGTERM"] as const) {
    process.on(signal, async () => {
      await supervisor.shutdown();
      stopOmp();
      process.exit(0);
    });
  }
}
