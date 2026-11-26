/**
 * Private session state, pin and profile marker (M19, M20, C6, G5) over a Host's sidecar operations.
 *
 * - Consumes: `Host.readState` / `Host.writeState`
 * - Provides: revisioned, per-session serialized `updateState`, `isPinned`, `copyPrivateState`, `profileOf`,
 *   `markProfile`, `isPlainObject`
 */
import type { Host, PrivateState, Profile } from "../host/index.ts";

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

/** Copy private data and the profile marker to a forked session; the fork starts unpinned at revision 0. */
export async function copyPrivateState(host: Host, fromPath: string, toPath: string): Promise<void> {
  const { data, profile } = await host.readState(fromPath);
  if (Object.keys(data).length === 0 && profile === undefined) return;
  await host.writeState(toPath, { revision: 0, pinned: false, data, ...(profile ? { profile } : {}) });
}

/** The profile a session was started with (G5), `coding` without a marker. */
export async function profileOf(host: Host, sessionPath: string): Promise<Profile> {
  return (await host.readState(sessionPath)).profile ?? "coding";
}

/** Record the `PI_CACHE_RETENTION` a session's Pi started with; Pi's default (`short`, or unset) leaves no marker. */
export async function markCacheRetention(host: Host, sessionPath: string, retention: string | undefined): Promise<void> {
  const marked = retention && retention !== "short" ? retention : undefined;
  await updateState(host, sessionPath, (state) => {
    if (state.cacheRetention === marked) return { result: undefined };
    const { cacheRetention: _previous, ...rest } = state;
    return { next: marked ? { ...rest, cacheRetention: marked } : rest, result: undefined };
  });
}

/** Record the profile a session runs with; no sidecar is written for an unmarked `coding` session. */
export async function markProfile(host: Host, sessionPath: string, profile: Profile): Promise<void> {
  await updateState(host, sessionPath, (state) => ({
    next: (state.profile ?? "coding") === profile ? undefined : { ...state, profile },
    result: undefined,
  }));
}
