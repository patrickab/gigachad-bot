/**
 * Private state sidecar files `<session>.jsonl.gigachad.json` on this machine (M19, M20, C6).
 *
 * - Consumes: session file paths
 * - Provides: `readState`, atomic `writeState`, `moveSidecar` for archive and restore, `removeSidecar` for deletion
 */
import { copyFile, mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { PROFILES, type PrivateState, type Profile } from "../index.ts";

/** Sidecar path of a session file. */
function sidecarPath(sessionPath: string): string {
  return `${sessionPath}.gigachad.json`;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * A session's private state, defaults when no sidecar exists or it is not JSON. Other read errors (EMFILE, EIO,
 * EACCES) throw: a default would unpin the session for eviction and be written back over the sidecar.
 */
export async function readState(sessionPath: string): Promise<PrivateState> {
  const defaults = { revision: 0, pinned: false, data: {} };
  let text: string;
  try {
    text = await readFile(sidecarPath(sessionPath), "utf8");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return defaults;
    throw error;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return defaults;
  }
  const record = isPlainObject(parsed) ? parsed : {};
  return {
    revision: Number.isInteger(record.revision) ? (record.revision as number) : 0,
    pinned: record.pinned === true,
    data: isPlainObject(record.data) ? record.data : {},
    ...(PROFILES.includes(record.profile as Profile) ? { profile: record.profile as Profile } : {}),
    ...(typeof record.cacheRetention === "string" ? { cacheRetention: record.cacheRetention } : {}),
  };
}

/** Write the sidecar through a temp file and rename so readers never see a partial file. */
export async function writeState(sessionPath: string, state: PrivateState): Promise<void> {
  const path = sidecarPath(sessionPath);
  await mkdir(dirname(path), { recursive: true });
  const temp = `${path}.${process.pid}.${crypto.randomUUID()}.tmp`;
  try {
    await writeFile(temp, JSON.stringify(state), { mode: 0o600 });
    await rename(temp, path);
  } catch (error) {
    await rm(temp, { force: true });
    throw error;
  }
}

/** Move the sidecar along with its session file, no-op when none exists. */
export async function moveSidecar(fromPath: string, toPath: string): Promise<void> {
  try {
    await rename(sidecarPath(fromPath), sidecarPath(toPath));
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return;
    if ((error as NodeJS.ErrnoException).code !== "EXDEV") throw error;
    await copyFile(sidecarPath(fromPath), sidecarPath(toPath));
    await rm(sidecarPath(fromPath));
  }
}

/** Remove the sidecar of a deleted session file, no-op when none exists. */
export async function removeSidecar(sessionPath: string): Promise<void> {
  await rm(sidecarPath(sessionPath), { force: true });
}
