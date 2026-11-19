/**
 * Private session state and pin (M19, M20, C6) in the sidecar `<session>.jsonl.gigachad.json`.
 *
 * - Consumes: session file paths
 * - Provides: revisioned read/update of `{revision, pinned, data}`, atomic JSON writes, sidecar copy and move
 */
import { copyFile, mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

export type PrivateState = { revision: number; pinned: boolean; data: Record<string, unknown> };

const sidecarLocks = new Map<string, Promise<unknown>>();

/** Sidecar path of a session file. */
export function sidecarPath(sessionPath: string): string {
  return `${sessionPath}.gigachad.json`;
}

/** True for a plain JSON object (not null, not an array). */
export function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Write JSON through a temp file and rename so readers never see a partial file. */
export async function writeJsonAtomic(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const temp = `${path}.${process.pid}.${crypto.randomUUID()}.tmp`;
  try {
    await writeFile(temp, JSON.stringify(value), { mode: 0o600 });
    await rename(temp, path);
  } catch (error) {
    await rm(temp, { force: true });
    throw error;
  }
}

/**
 * Read a session's private state, defaults when no sidecar exists or it is not JSON. Other read errors (EMFILE, EIO,
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
  };
}

/** Read-modify-write the sidecar, serialized per session so concurrent requests cannot lose updates. */
export async function updateState<T>(
  sessionPath: string,
  update: (state: PrivateState) => { next?: PrivateState; result: T },
): Promise<T> {
  const key = sidecarPath(sessionPath);
  const previous = sidecarLocks.get(key) ?? Promise.resolve();
  const run = previous.catch(() => {}).then(async () => {
    const { next, result } = update(await readState(sessionPath));
    if (next) await writeJsonAtomic(key, next);
    return result;
  });
  sidecarLocks.set(key, run);
  try {
    return await run;
  } finally {
    if (sidecarLocks.get(key) === run) sidecarLocks.delete(key);
  }
}

/** True when the session's private state pins it (M20). */
export async function isPinned(sessionPath: string): Promise<boolean> {
  return (await readState(sessionPath)).pinned;
}

/** Copy private data to a forked session; the fork starts unpinned at revision 0. */
export async function copyPrivateState(fromPath: string, toPath: string): Promise<void> {
  const { data } = await readState(fromPath);
  if (Object.keys(data).length === 0) return;
  await writeJsonAtomic(sidecarPath(toPath), { revision: 0, pinned: false, data });
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
