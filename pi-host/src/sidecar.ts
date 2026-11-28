/**
 * Private session state, pin and profile marker (M19, M20, C6, G5) over a Host's sidecar operations.
 *
 * - Consumes: `Host.readState` / `Host.writeState`
 * - Provides: per-session serialized `updateState`, `isPinned`, `copyPrivateState`, `profileOf`, `markSpawn`
 */
import type { Host, PrivateState, Profile } from "./host/index.ts";

const sidecarLocks = new Map<string, Promise<unknown>>();

/** Read-modify-write the state, serialized per session so concurrent requests cannot lose updates. */
export async function updateState(host: Host, sessionPath: string, update: (state: PrivateState) => PrivateState | undefined): Promise<void> {
  const key = `${host.machine}:${sessionPath}`;
  const previous = sidecarLocks.get(key) ?? Promise.resolve();
  const run = previous.catch(() => {}).then(async () => {
    const next = update(await host.readState(sessionPath));
    if (next) await host.writeState(sessionPath, next);
  });
  sidecarLocks.set(key, run);
  try {
    await run;
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

/**
 * Record the profile and the `PI_CACHE_RETENTION` a session's Pi started with. A `coding` session with Pi's default
 * retention (`short`, or unset) gets no sidecar.
 */
export async function markSpawn(host: Host, sessionPath: string, profile: Profile, retention: string | undefined): Promise<void> {
  const marked = retention && retention !== "short" ? retention : undefined;
  await updateState(host, sessionPath, ({ cacheRetention, ...state }) => {
    const sameProfile = (state.profile ?? "coding") === profile;
    if (sameProfile && cacheRetention === marked) return undefined;
    return { ...state, ...(sameProfile ? {} : { profile }), ...(marked ? { cacheRetention: marked } : {}) };
  });
}
