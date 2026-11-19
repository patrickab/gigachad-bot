/**
 * Private session state and pin (M19, M20, C6) over a Host's sidecar operations.
 *
 * - Consumes: `Host.readState` / `Host.writeState`
 * - Provides: revisioned, per-session serialized `updateState`, `isPinned`, `copyPrivateState`, `isPlainObject`
 */
import type { Host, PrivateState } from "../host/index.ts";

const sidecarLocks = new Map<string, Promise<unknown>>();

/** True for a plain JSON object (not null, not an array). */
export function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Read-modify-write the state, serialized per session so concurrent requests cannot lose updates. */
export async function updateState<T>(
  host: Host,
  sessionPath: string,
  update: (state: PrivateState) => { next?: PrivateState; result: T },
): Promise<T> {
  const key = `${host.machine}:${sessionPath}`;
  const previous = sidecarLocks.get(key) ?? Promise.resolve();
  const run = previous.catch(() => {}).then(async () => {
    const { next, result } = update(await host.readState(sessionPath));
    if (next) await host.writeState(sessionPath, next);
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
export async function isPinned(host: Host, sessionPath: string): Promise<boolean> {
  return (await host.readState(sessionPath)).pinned;
}

/** Copy private data to a forked session; the fork starts unpinned at revision 0. */
export async function copyPrivateState(host: Host, fromPath: string, toPath: string): Promise<void> {
  const { data } = await host.readState(fromPath);
  if (Object.keys(data).length === 0) return;
  await host.writeState(toPath, { revision: 0, pinned: false, data });
}
