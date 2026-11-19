/**
 * The M1 lock a live session holds on its session file, taken and released through the `Host`.
 *
 * - Consumes: `Host.lockSession` / `Host.unlockSession`
 * - Provides: `SessionLock`; a resumed session takes it before its child starts, a new one once Pi wrote the file
 */
import type { Host } from "../host/index.ts";

export type TakeOutcome = "held" | "busy" | "missing";

export class SessionLock {
  private lockId: string | undefined;
  private released = false;

  constructor(private readonly host: Host) {}

  get held(): boolean {
    return this.lockId !== undefined;
  }

  /** Lock `sessionPath`; a lock that arrives after `release` is given back at once. */
  async take(sessionPath: string): Promise<TakeOutcome> {
    if (this.lockId) return "held";
    const result = await this.host.lockSession(sessionPath);
    if (!result.ok) return result.error;
    if (this.released) {
      void this.host.unlockSession(result.lockId).catch(() => undefined);
      return "busy";
    }
    this.lockId = result.lockId;
    return "held";
  }

  release(): void {
    this.released = true;
    if (this.lockId) void this.host.unlockSession(this.lockId).catch(() => undefined);
    this.lockId = undefined;
  }
}
