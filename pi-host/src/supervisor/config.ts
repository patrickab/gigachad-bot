/**
 * Supervisor configuration from the environment and the pinned Pi version (M13).
 *
 * - Consumes: `PI_HOST_PI_BIN`, `PI_HOST_IDLE_MINUTES`, `PI_HOST_CHILD_ENV`,
 *   the `@earendil-works/pi-coding-agent` pin in `package.json`
 * - Provides: `SupervisorConfig`, `loadConfig`, `PINNED_PI_VERSION`
 */
import packageJson from "../../package.json";

/** The Pi version pi-host speaks to, the same pin as the SDK dependency. */
export const PINNED_PI_VERSION: string = packageJson.dependencies["@earendil-works/pi-coding-agent"];

const DEFAULT_IDLE_MINUTES = 15;

export type SupervisorConfig = {
  /** Pi executable, `pi` on PATH unless `PI_HOST_PI_BIN` is set (C4). */
  piBin: string;
  /** A session with no client, not running and not pinned is stopped after this long. */
  idleMs: number;
  /** Extra variable names children inherit (`PI_HOST_CHILD_ENV`, comma separated). */
  extraChildEnv: string[];
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
  };
}
