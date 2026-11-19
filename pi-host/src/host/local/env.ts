/**
 * The environment a local Pi child inherits (M6): an allowlist over pi-host's own environment.
 *
 * - Consumes: pi-host's environment, extra names from `PI_HOST_CHILD_ENV`
 * - Provides: `childEnv`
 */

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

/** The allowlisted environment a Pi child gets. Pi's own update check is off, the version is pinned. */
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
