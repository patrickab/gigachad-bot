/**
 * Session supervisor: one `pi --mode rpc` child per live session, attach with replay, many clients, idle eviction.
 *
 * - Consumes: a `Host` (spawn, session lock, session metadata; Pi version checked at start, M13), Catalog's helpers
 * - Provides: `createSupervisor()`, the `Supervisor` HTTP rows (`GET /health` with live sessions, `POST /sessions`,
 *   `POST /sessions/:id/close`), the multiplexed `/ws` handler (M16, M18, resync M2) and `LiveSessions` (live status,
 *   closing an idle child before archive) for the catalog
 *
 * Liveness (M17): Pi exposes no liveness registry over RPC, so an extension with background work keeps its session
 * from idle eviction by setting a non-empty `setStatus` under key `session-liveness` and clearing it when done.
 */
import type { WebSocketHandler } from "bun";
import { isPinned } from "../catalog/index.ts";
import type { Host } from "../host/index.ts";
import { LocalHost } from "../host/local/index.ts";
import { loadConfig, PINNED_PI_VERSION, type SupervisorConfig } from "./config.ts";
import { SupervisorHost } from "./host.ts";

export type SocketData = { clientId: string };

/** One live session in `GET /health`. */
export type LiveSessionHealth = { sessionId: string; sessionPath: string; running: boolean; attached: number; pinned: boolean };
export type Health = { ok: true; piVersion: string; sessions: LiveSessionHealth[] };

/** What the catalog needs to know about running sessions. */
export interface LiveSessions {
  /** Live status of a session file, undefined when no child holds it. */
  liveStatus(sessionPath: string): { running: boolean; waiting: boolean; attached: number } | undefined;
  /** Close the child holding a session file when nothing needs it now (no run, client, dialog or liveness status). True once no child holds it. */
  closeIfIdle(sessionPath: string): Promise<boolean>;
}

export interface Supervisor extends LiveSessions {
  /** The machine the sessions run on, shared with the catalog. */
  readonly host: Host;
  socketData(): SocketData;
  websocket: WebSocketHandler<SocketData>;
  handle(request: Request, url: URL): Promise<Response | undefined>;
  health(): Promise<Health>;
  shutdown(): Promise<void>;
}

export type SupervisorOptions = Partial<SupervisorConfig> & {
  /** Machine seam (M23), a `LocalHost` over the config by default. */
  host?: Host;
  /** Pin lookup (M20), Catalog's sidecar by default. */
  isPinned?: (sessionPath: string) => Promise<boolean>;
};

/** Check the host's Pi binary and start the supervisor. Throws unless it is the pinned version (M13). */
export async function createSupervisor(options: SupervisorOptions = {}): Promise<Supervisor> {
  const { host: givenHost, isPinned: pinLookup, ...overrides } = options;
  const config = { ...loadConfig(), ...overrides };
  const host = givenHost ?? new LocalHost(config);
  const version = await host.piVersion();
  if (version !== PINNED_PI_VERSION) {
    throw new Error(`Pi on ${host.machine} reported "${version}", pi-host needs ${PINNED_PI_VERSION}`);
  }
  return new SupervisorHost(config, version, pinLookup ?? ((sessionPath) => isPinned(host, sessionPath)), host);
}
