/**
 * `LocalHost`: the `Host` of the machine pi-host runs on (machine id `local`).
 *
 * - Consumes: the Pi binary, pi-host's environment, Pi's session files and sidecars, git in a requested cwd
 * - Provides: `LocalHost`, the only place that spawns processes, touches or watches files or takes `flock`s
 *
 * Spawn strategies (plan D5): `coding` and `document` run the Pi binary directly, the profile shapes only its
 * arguments (`supervisor/profiles.ts`). A sandboxed profile later maps to a strategy that runs
 * `agent-sandbox pi-rpc --profile ... -- <pi args>`, which keeps the same stdio, stdin-EOF and SIGTERM contract.
 */
import { stat } from "node:fs/promises";
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
import { searchSessions } from "./search.ts";
import { findSession, listSessions, moveSession, sessionMeta } from "./sessions.ts";
import { readState, writeState } from "./state.ts";
import { watchSessions } from "./watch.ts";

export type LocalHostOptions = {
  /** Pi executable (C4). */
  piBin: string;
  /** Extra variable names children inherit beyond the M6 allowlist. */
  extraChildEnv: string[];
};

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

  async spawn({ cwd, piArgs }: SpawnRequest): Promise<PiChild> {
    // Every profile runs the Pi binary directly.
    return Bun.spawn([this.options.piBin, ...piArgs], {
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
