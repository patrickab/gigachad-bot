/**
 * Session supervisor: one `pi --mode rpc` child per live session, attach with replay, many clients, idle eviction.
 *
 * - Consumes: the pinned `pi` binary (checked at start, M13), Pi's session files, Catalog's session helpers
 * - Provides: `createSupervisor()`, the `Supervisor` HTTP rows (`POST /sessions`, `POST /sessions/:id/close`),
 *   the multiplexed `/ws` handler (M16, M18) and `LiveSessions.liveStatus` for the catalog
 *
 * Liveness (M17): Pi exposes no liveness registry over RPC, so an extension with background work keeps its session
 * from idle eviction by setting a non-empty `setStatus` under key `session-liveness` and clearing it when done.
 */
import type { WebSocketHandler } from "bun";
import { isPinned } from "../catalog/index.ts";
import { childEnv, loadConfig, PINNED_PI_VERSION, type SupervisorConfig } from "./config.ts";
import { SupervisorHost } from "./host.ts";

export type SocketData = { clientId: string };

/** What the catalog needs to know about running sessions. */
export interface LiveSessions {
  /** Live status of a session file, undefined when no child holds it. */
  liveStatus(sessionPath: string): { running: boolean; waiting: boolean; attached: number } | undefined;
}

export interface Supervisor extends LiveSessions {
  socketData(): SocketData;
  websocket: WebSocketHandler<SocketData>;
  handle(request: Request, url: URL): Promise<Response | undefined>;
  health(): { ok: true; piVersion: string };
  shutdown(): Promise<void>;
}

export type SupervisorOptions = Partial<SupervisorConfig> & {
  /** Pin lookup (M20), Catalog's sidecar by default. */
  isPinned?: (sessionPath: string) => Promise<boolean>;
};

/** Version the Pi binary reports, rejected unless it is the pinned one (M13). */
async function checkPiVersion(config: SupervisorConfig): Promise<string> {
  const child = Bun.spawn([config.piBin, "--version"], {
    env: childEnv(config.extraChildEnv),
    stdout: "pipe",
    stderr: "pipe",
  });
  const [output, code] = await Promise.all([new Response(child.stdout).text(), child.exited]);
  const version = output.trim().split(/\s+/).pop() ?? "";
  if (code !== 0 || version !== PINNED_PI_VERSION) {
    throw new Error(`${config.piBin} --version reported "${output.trim()}" (exit ${code}), pi-host needs ${PINNED_PI_VERSION}`);
  }
  return version;
}

/** Check the Pi binary and start the supervisor. Throws on a version mismatch. */
export async function createSupervisor(options: SupervisorOptions = {}): Promise<Supervisor> {
  const { isPinned: pinLookup = isPinned, ...overrides } = options;
  const config = { ...loadConfig(), ...overrides };
  return new SupervisorHost(config, await checkPiVersion(config), pinLookup);
}
