/**
 * Who owns a process id, safe against pid reuse: a pid alone says nothing after a restart or reboot.
 *
 * - Consumes: `/proc/<pid>/stat` (start time, field 22) and the boot id (Linux, pi-host's only platform: `flock` and
 *   `bun:ffi` locking are Linux-specific as well)
 * - Provides: `isProcessAlive`, `processStartIdentity`
 */
import { readFile } from "node:fs/promises";

/** `pid` names a process that exists (a zombie counts, the identity check tells the owner apart). */
export function isProcessAlive(pid: number): boolean {
  if (!Number.isSafeInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    // EPERM: it exists and belongs to another user.
    return (error as NodeJS.ErrnoException).code === "EPERM";
  }
}

/** Kernel start time of `pid` qualified by the boot id, null when the process is gone or the platform has no `/proc`. */
export async function processStartIdentity(pid: number): Promise<string | null> {
  if (!Number.isSafeInteger(pid) || pid <= 0) return null;
  try {
    const stat = await readFile(`/proc/${pid}/stat`, "utf8");
    // After the parenthesised command name (it may contain spaces): state is field 3, start time field 22.
    const start = stat.slice(stat.lastIndexOf(")") + 2).split(" ")[19];
    if (!/^\d+$/.test(start ?? "")) return null;
    const boot = (await readFile("/proc/sys/kernel/random/boot_id", "utf8")).trim();
    return `linux:${boot}:${start}`;
  } catch {
    return null;
  }
}
