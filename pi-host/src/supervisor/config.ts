/**
 * Supervisor configuration from the environment, and the Pi SDK version pi-host is built against.
 *
 * - Consumes: `PI_HOST_PI_BIN`, `PI_HOST_IDLE_MINUTES`, `PI_HOST_CHILD_ENV`, `PI_HOST_PORT`, `PI_HOST_EXTENSIONS_DIR`,
 *   `PI_HOST_AGENT_SANDBOX_BIN`, `PI_HOST_SANDBOX_PROFILE`, the gateway token path (`auth.ts`),
 *   the `@earendil-works/pi-coding-agent` pin in `package.json`
 * - Provides: `SupervisorConfig`, `loadConfig`, `PINNED_PI_VERSION`
 */
import { resolve } from "node:path";
import packageJson from "../../package.json";
import { gatewayTokenPath } from "../auth.ts";

/** The version of the Pi SDK dependency. The installed Pi binary may be newer: pi-host runs whatever the host has. */
export const PINNED_PI_VERSION: string = packageJson.dependencies["@earendil-works/pi-coding-agent"];

const DEFAULT_IDLE_MINUTES = 15;
const DEFAULT_PORT = 8300;

export type SupervisorConfig = {
  /** Pi executable, `pi` on PATH unless `PI_HOST_PI_BIN` is set (C4). */
  piBin: string;
  /** A session with no client, not running and not pinned is stopped after this long. */
  idleMs: number;
  /** Extra variable names children inherit (`PI_HOST_CHILD_ENV`, comma separated). */
  extraChildEnv: string[];
  /** pi-host's own port (`PI_HOST_PORT`), where sandboxed children reach the completion gateway. */
  gatewayPort: number;
  /** The gateway-only token file (C1) handed to sandboxed children. */
  gatewayTokenPath: string;
  /** Repo extensions loaded into unrestricted local children (C6, `PI_HOST_EXTENSIONS_DIR`, default `<repo>/pi-extensions`). */
  extensionsDir: string;
  /** Agents-in-a-box CLI running `sandboxed` children (C5, `PI_HOST_AGENT_SANDBOX_BIN`, default `agent-sandbox`). */
  agentSandboxBin: string;
  /** AiB profile of `sandboxed` children (`PI_HOST_SANDBOX_PROFILE`, default `gigachad`). */
  sandboxProfile: string;
};

/** Configuration from pi-host's environment. */
export function loadConfig(env: NodeJS.ProcessEnv = process.env): SupervisorConfig {
  const idleMinutes = Number(env.PI_HOST_IDLE_MINUTES ?? DEFAULT_IDLE_MINUTES);
  return {
    piBin: env.PI_HOST_PI_BIN || "pi",
    idleMs: (Number.isFinite(idleMinutes) && idleMinutes > 0 ? idleMinutes : DEFAULT_IDLE_MINUTES) * 60_000,
    extraChildEnv: (env.PI_HOST_CHILD_ENV ?? "")
      .split(",")
      .map((name) => name.trim())
      .filter(Boolean),
    gatewayPort: Number(env.PI_HOST_PORT || DEFAULT_PORT),
    gatewayTokenPath,
    extensionsDir: resolve(env.PI_HOST_EXTENSIONS_DIR || resolve(import.meta.dir, "../../../pi-extensions")),
    agentSandboxBin: env.PI_HOST_AGENT_SANDBOX_BIN || "agent-sandbox",
    sandboxProfile: env.PI_HOST_SANDBOX_PROFILE || "gigachad",
  };
}
