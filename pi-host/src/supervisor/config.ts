/**
 * Supervisor configuration from the environment, the pinned Pi version (M13) and the child env allowlist (M6).
 *
 * - Consumes: `PI_HOST_PI_BIN`, `PI_HOST_IDLE_MINUTES`, `PI_HOST_CHILD_ENV`, pi-host's own environment,
 *   the `@earendil-works/pi-coding-agent` pin in `package.json`
 * - Provides: `SupervisorConfig`, `loadConfig`, `PINNED_PI_VERSION`, `childEnv`
 */
import packageJson from "../../package.json";

/** The Pi version pi-host speaks to, the same pin as the SDK dependency. */
export const PINNED_PI_VERSION: string = packageJson.dependencies["@earendil-works/pi-coding-agent"];

const DEFAULT_IDLE_MINUTES = 15;

/** Exact variable names a child may inherit. Provider keys and backend secrets are not among them. */
const CHILD_ENV_NAMES = new Set([
  "HOME",
  "USER",
  "LOGNAME",
  "SHELL",
  "PATH",
  "LANG",
  "LANGUAGE",
  "TERM",
  "COLORTERM",
  "TZ",
  "TMPDIR",
  "EDITOR",
  "VISUAL",
  "SSH_AUTH_SOCK",
  "HTTP_PROXY",
  "HTTPS_PROXY",
  "NO_PROXY",
  "http_proxy",
  "https_proxy",
  "no_proxy",
  "PI_CODING_AGENT_DIR",
  "PI_CODING_AGENT_SESSION_DIR",
  "PI_PACKAGE_DIR",
  "PI_OFFLINE",
  "PI_TELEMETRY",
  "PI_CACHE_RETENTION",
]);
const CHILD_ENV_PREFIXES = ["LC_", "XDG_"];

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

/** The allowlisted environment a Pi child gets (M6). Pi's own update check is off, the version is pinned. */
export function childEnv(extraNames: string[], env: NodeJS.ProcessEnv = process.env): Record<string, string> {
  const allowed = new Set([...CHILD_ENV_NAMES, ...extraNames]);
  const result: Record<string, string> = {};
  for (const [name, value] of Object.entries(env)) {
    if (value === undefined) continue;
    if (allowed.has(name) || CHILD_ENV_PREFIXES.some((prefix) => name.startsWith(prefix))) result[name] = value;
  }
  result.PI_SKIP_VERSION_CHECK = "1";
  return result;
}
