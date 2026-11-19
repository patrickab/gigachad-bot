/**
 * `LocalHost`: the `Host` of the machine pi-host runs on (machine id `local`).
 *
 * - Consumes: the Pi binary, the `agent-sandbox` CLI, pi-host's environment, the repo extensions dir, Pi's session
 *   files and sidecars, git in a requested cwd
 * - Provides: `LocalHost`, the only place that spawns processes, touches or watches files or takes `flock`s
 *
 * Spawn strategies (plan D5): `coding` and `document` run the Pi binary directly, the profile shapes its arguments
 * (`supervisor/profiles.ts`) and `coding` also loads the repo extensions (C6). `sandboxed` runs the same arguments
 * through `agent-sandbox pi-rpc` (`sandbox.ts`, C4, C5), which keeps the same stdio, stdin-EOF and SIGTERM contract.
 */
import type { Dirent } from "node:fs";
import { readdir, stat } from "node:fs/promises";
import { join } from "node:path";
import type {
  ForkRequest,
  ForkResult,
  Host,
  LockResult,
  MoveResult,
  PiChild,
  PrivateState,
  SearchMatch,
  SessionMeta,
  SpawnRequest,
} from "../index.ts";
import { childEnv } from "./env.ts";
import { listFiles } from "./files.ts";
import { forkSession } from "./fork.ts";
import { tryLockFile, type FileLock } from "./lock.ts";
import { spawnSandboxed } from "./sandbox.ts";
import { searchSessions } from "./search.ts";
import { findSession, listSessions, moveSession, sessionMeta } from "./sessions.ts";
import { readState, writeState } from "./state.ts";
import { watchSessions } from "./watch.ts";

export type LocalHostOptions = {
  /** Pi executable (C4). */
  piBin: string;
  /** Extra variable names children inherit beyond the M6 allowlist. */
  extraChildEnv: string[];
  /** Repo extensions dir, each top-level extension loaded into `coding` children (C6). */
  extensionsDir: string;
  /** `agent-sandbox` CLI running `sandboxed` children (C5). */
  agentSandboxBin: string;
  /** AiB profile of `sandboxed` children. */
  sandboxProfile: string;
  /** pi-host's own port, the completion gateway sandboxed children reach. */
  gatewayPort: number;
  /** The gateway-only token file (C1), the only credential a sandboxed child gets. */
  gatewayTokenPath: string;
};

/** True for a C6 extension entry: a `*.ts` file (not `*.test.ts` or `*.d.ts`) or a directory holding `index.ts`. */
async function isExtension(dir: string, entry: Dirent): Promise<boolean> {
  if (entry.isFile()) return entry.name.endsWith(".ts") && !/\.(test|d)\.ts$/.test(entry.name);
  if (!entry.isDirectory()) return false;
  return (await stat(join(dir, entry.name, "index.ts")).catch(() => undefined))?.isFile() ?? false;
}

/** `-e <abs path>` for each extension in `dir`, sorted, listed per spawn; none when `dir` is missing. */
async function extensionArgs(dir: string): Promise<string[]> {
  const entries = await readdir(dir, { withFileTypes: true }).catch((error: NodeJS.ErrnoException) => {
    if (error.code === "ENOENT") return [];
    throw error;
  });
  const kept = await Promise.all(entries.map(async (entry) => ((await isExtension(dir, entry)) ? entry.name : undefined)));
  return kept
    .filter((name): name is string => name !== undefined)
    .sort()
    .flatMap((name) => ["-e", join(dir, name)]);
}

export class LocalHost implements Host {
  readonly machine = "local";
  private readonly locks = new Map<string, FileLock>();

  constructor(private readonly options: LocalHostOptions) {}

  async piVersion(): Promise<string> {
    const child = Bun.spawn([this.options.piBin, "--version"], {
      env: childEnv(this.options.extraChildEnv),
      stdout: "pipe",
      stderr: "pipe",
    });
    const [output, code] = await Promise.all([new Response(child.stdout).text(), child.exited]);
    if (code !== 0) throw new Error(`${this.options.piBin} --version exited ${code}: "${output.trim()}"`);
    return output.trim().split(/\s+/).pop() ?? "";
  }

  async spawn({ profile, cwd, piArgs }: SpawnRequest): Promise<PiChild> {
    if (profile === "sandboxed") return spawnSandboxed(this.options, cwd, piArgs);
    const extensions = profile === "coding" ? await extensionArgs(this.options.extensionsDir) : [];
    return Bun.spawn([this.options.piBin, ...piArgs, ...extensions], {
      cwd,
      env: childEnv(this.options.extraChildEnv),
      stdin: "pipe",
      stdout: "pipe",
      stderr: "pipe",
    });
  }

  async lockSession(sessionPath: string): Promise<LockResult> {
    let lock: FileLock | undefined;
    try {
      lock = tryLockFile(sessionPath);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return { ok: false, error: "missing" };
      throw error;
    }
    if (!lock) return { ok: false, error: "busy" };
    const lockId = crypto.randomUUID();
    this.locks.set(lockId, lock);
    return { ok: true, lockId };
  }

  async unlockSession(lockId: string): Promise<void> {
    this.locks.get(lockId)?.release();
    this.locks.delete(lockId);
  }

  async isDirectory(path: string): Promise<boolean> {
    return (await stat(path).catch(() => undefined))?.isDirectory() ?? false;
  }

  listSessions(cwd: string | null, archived: boolean): Promise<SessionMeta[]> {
    return listSessions(cwd, archived);
  }

  sessionMeta(sessionPath: string): Promise<SessionMeta | undefined> {
    return sessionMeta(sessionPath);
  }

  findSession(sessionId: string): Promise<string | undefined> {
    return findSession(sessionId);
  }

  moveSession(sessionPath: string, toArchive: boolean): Promise<MoveResult> {
    return moveSession(sessionPath, toArchive);
  }

  forkSession(request: ForkRequest): Promise<ForkResult> {
    return forkSession(request);
  }

  readState(sessionPath: string): Promise<PrivateState> {
    return readState(sessionPath);
  }

  writeState(sessionPath: string, state: PrivateState): Promise<void> {
    return writeState(sessionPath, state);
  }

  listFiles(cwd: string): Promise<string[]> {
    return listFiles(cwd);
  }

  async searchSessions(query: string, limit: number): Promise<SearchMatch[]> {
    return searchSessions(query, limit, await listSessions(null, false));
  }

  watchSessions(onChange: (sessionPaths: string[]) => void): () => void {
    return watchSessions(onChange);
  }
}
