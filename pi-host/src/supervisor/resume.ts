/**
 * Auto-resume of interrupted runs: a small state file per pi-host process lists the sessions with an active run, the
 * next pi-host start claims the lists of dead owners and resumes those sessions (opt-in).
 *
 * - Consumes: pi-host's state directory, `process-identity.ts`
 * - Provides: `ResumeState` (`track` on run start, `untrack` on settle, `claimOrphans` on start, the `autoResume`
 *   setting), `RESUME_PROMPT`
 *
 * Files in the state directory:
 * - `active-runs-<pid>.json`: `{ownerPid, ownerStart, sessions: [{sessionId, sessionPath, cwd, spec}]}`, written
 *   atomically (temp file, rename), deleted when it would be empty
 * - `pi-host-settings.json`: `{autoResume}`, off by default
 *
 * An owner is dead when its pid is gone or its start identity (`/proc` start time + boot id) differs from the recorded
 * one, so a reused pid or a reboot never protects a stale list. A claim renames the list to a name of this process
 * first: of two pi-hosts starting together only one wins the rename, and a list is resumed once.
 */
import { randomUUID } from "node:crypto";
import { mkdir, readdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import type { ProfileSpec } from "./profiles.ts";
import { isProcessAlive, processStartIdentity } from "./process-identity.ts";

export const RESUME_PROMPT = "The session was interrupted by a restart. Continue where you left off.";

const LIST_FILE = /^active-runs-(\d+)\.json$/;

export type RunRecord = { sessionId: string; sessionPath: string; cwd: string; spec: ProfileSpec };
type RunList = { ownerPid: number; ownerStart: string | null; sessions: RunRecord[] };

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseList(text: string): RunList | undefined {
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    return undefined;
  }
  if (!isRecord(data) || typeof data.ownerPid !== "number" || !Array.isArray(data.sessions)) return undefined;
  const sessions = data.sessions.filter(
    (entry): entry is RunRecord =>
      isRecord(entry) &&
      typeof entry.sessionId === "string" &&
      typeof entry.sessionPath === "string" &&
      typeof entry.cwd === "string" &&
      isRecord(entry.spec),
  );
  return { ownerPid: data.ownerPid, ownerStart: typeof data.ownerStart === "string" ? data.ownerStart : null, sessions };
}

export class ResumeState {
  private readonly active = new Map<string, RunRecord>();
  private writes: Promise<unknown> = Promise.resolve();
  private claimed: Promise<RunRecord[]> | undefined;
  private ownStart: Promise<string | null> | undefined;

  constructor(
    private readonly dir: string,
    private readonly pid = process.pid,
    private readonly identityOf: (pid: number) => Promise<string | null> = processStartIdentity,
    private readonly aliveOf: (pid: number) => boolean = isProcessAlive,
  ) {}

  private get path(): string {
    return join(this.dir, `active-runs-${this.pid}.json`);
  }

  private get settingsPath(): string {
    return join(this.dir, "pi-host-settings.json");
  }

  /** Whether interrupted runs are resumed on start, off until the user turns it on. */
  async autoResume(): Promise<boolean> {
    try {
      const data: unknown = JSON.parse(await readFile(this.settingsPath, "utf8"));
      return isRecord(data) && data.autoResume === true;
    } catch {
      return false;
    }
  }

  async setAutoResume(enabled: boolean): Promise<void> {
    await this.writeAtomic(this.settingsPath, JSON.stringify({ autoResume: enabled }));
  }

  /** A run started in `record`'s session. */
  track(record: RunRecord): Promise<void> {
    this.active.set(record.sessionId, record);
    return this.persist();
  }

  /** The session's run settled, was aborted or its child ended. */
  untrack(sessionId: string): Promise<void> {
    if (!this.active.delete(sessionId)) return this.writes.then(() => undefined);
    return this.persist();
  }

  /** Wait for every queued write. */
  async flush(): Promise<void> {
    await this.writes;
  }

  /**
   * Take the lists of dead owners, once per process: the records of their runs. Lists of live owners stay. Runs before
   * this process writes its own list, which may carry the pid of a dead owner from before a reboot.
   */
  claimOrphans(): Promise<RunRecord[]> {
    this.claimed ??= this.claim();
    return this.claimed;
  }

  private async claim(): Promise<RunRecord[]> {
    let names: string[];
    try {
      names = await readdir(this.dir);
    } catch {
      return [];
    }
    const ownStart = await this.ownIdentity();
    const records: RunRecord[] = [];
    for (const name of names) {
      const owner = LIST_FILE.exec(name);
      if (!owner) continue;
      const path = join(this.dir, name);
      const taken = `${path}.claimed-${this.pid}-${randomUUID()}`;
      let list: RunList | undefined;
      try {
        list = parseList(await readFile(path, "utf8"));
      } catch {
        continue;
      }
      if (list && (await this.isLive(list, ownStart))) continue;
      try {
        await rename(path, taken);
      } catch {
        // Another pi-host won the claim, or the list is gone.
        continue;
      }
      try {
        // Re-read what was claimed: the owner may have rewritten the list between check and rename.
        list = parseList(await readFile(taken, "utf8"));
        if (list) records.push(...list.sessions);
      } finally {
        await rm(taken, { force: true });
      }
    }
    return records;
  }

  /** The list's owner is still running: same pid and the same start identity (an unrecorded one falls back to the pid). */
  private async isLive(list: RunList, ownStart: string | null): Promise<boolean> {
    if (list.ownerPid === this.pid && list.ownerStart === ownStart) return true;
    if (!this.aliveOf(list.ownerPid)) return false;
    if (list.ownerStart === null) return true;
    const current = await this.identityOf(list.ownerPid);
    // An unreadable identity of a live pid cannot prove reuse: keep the list.
    return current === null || current === list.ownerStart;
  }

  private ownIdentity(): Promise<string | null> {
    this.ownStart ??= this.identityOf(this.pid);
    return this.ownStart;
  }

  private persist(): Promise<void> {
    const write = this.writes.then(async () => {
      await this.claimOrphans();
      if (this.active.size === 0) {
        await rm(this.path, { force: true });
        return;
      }
      const list: RunList = { ownerPid: this.pid, ownerStart: await this.ownIdentity(), sessions: [...this.active.values()] };
      await this.writeAtomic(this.path, JSON.stringify(list));
    });
    this.writes = write.catch(() => undefined);
    return write;
  }

  private async writeAtomic(path: string, content: string): Promise<void> {
    await mkdir(this.dir, { recursive: true, mode: 0o700 });
    const temp = `${path}.${process.pid}.${randomUUID()}.tmp`;
    await writeFile(temp, content, { mode: 0o600 });
    await rename(temp, path);
  }
}
