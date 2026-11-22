/**
 * Where Pi keeps session files on this machine, resolved the way Pi 1.5.2 does.
 *
 * - Consumes: `PI_CODING_AGENT_DIR` (default `~/.pi/agent`) and `PI_CODING_AGENT_SESSION_DIR`
 * - Provides: the live and archive session roots (C7), the factory plugin's data root and its session root, and the
 *   directory holding one cwd's sessions
 */
import { homedir } from "node:os";
import { basename, dirname, join, relative, resolve, sep } from "node:path";

/** Expand a leading `~` and make the path absolute. */
function expandHome(path: string): string {
  if (path === "~") return homedir();
  if (path.startsWith("~/")) return join(homedir(), path.slice(2));
  return resolve(path);
}

/** Pi's agent config dir. */
export function agentDir(): string {
  const override = process.env.PI_CODING_AGENT_DIR;
  return override ? expandHome(override) : join(homedir(), ".pi", "agent");
}

/** Flat session dir override, undefined when Pi groups sessions per cwd. */
function sessionDirOverride(): string | undefined {
  const override = process.env.PI_CODING_AGENT_SESSION_DIR;
  return override ? expandHome(override) : undefined;
}

/** Root of the live session files. */
export function sessionsRoot(): string {
  return sessionDirOverride() ?? join(agentDir(), "sessions");
}

/** Root of archived session files, a sibling of the live root (`sessions` -> `sessions-archive`). */
export function archiveRoot(): string {
  const root = sessionsRoot();
  return join(dirname(root), `${basename(root)}-archive`);
}

/** The factory plugin's data (task records, task sessions, repository settings), always under the agent dir. */
export function factoryRoot(): string {
  return join(agentDir(), "factory");
}

/**
 * Root of the factory task sessions, which the factory plugin writes apart from the live root so `pi -r` and the
 * session picker never list them, also when `PI_CODING_AGENT_SESSION_DIR` overrides the live root.
 */
export function factorySessionsRoot(): string {
  return join(factoryRoot(), "sessions");
}

/** Pi's per-cwd directory name: leading separator dropped, `/`, `\` and `:` replaced by `-`. */
export function cwdDirName(cwd: string): string {
  return `--${resolve(cwd).replace(/^[/\\]/, "").replace(/[/\\:]/g, "-")}--`;
}

/** Directory holding the sessions of `cwd` under `root` (flat when the session dir is overridden). */
export function cwdSessionDir(root: string, cwd: string): string {
  return sessionDirOverride() ? root : join(root, cwdDirName(cwd));
}

/** True when `path` lies inside `root`. */
export function isInside(root: string, path: string): boolean {
  const rel = relative(root, path);
  return rel !== "" && !rel.startsWith(`..${sep}`) && rel !== ".." && !rel.startsWith(sep);
}
