/**
 * Pi's model runtime for the gateway and /usage: Pi's registry, `auth.json` and token refresh.
 *
 * - Consumes: the Pi agent dir (`PI_CODING_AGENT_DIR`, default `~/.pi/agent`): `auth.json`, `models.json`, `models-store.json`
 * - Provides: `reloadingPiRuntime`, a pi-ai `Models` collection shared with running Pi processes that reloads
 *   `models.json` when it changes (the backend syncs Ollama's models into it), so no restart is needed
 *
 * Writes: only `auth.json`, when an OAuth token is refreshed (under Pi's own file lock, so a rotated
 * refresh token is never lost to the Pi sessions sharing it). The model catalog store is read-only
 * here and catalogs are never refreshed over the network: Pi itself keeps `models-store.json` current.
 */
import { stat } from "node:fs/promises";
import { join } from "node:path";
import type { Models, ModelsStore, ModelsStoreEntry } from "@earendil-works/pi-ai";
import { getAgentDir, ModelRuntime } from "@earendil-works/pi-coding-agent";

/** Pi's `models-store.json`, read on demand, writes dropped. */
class ReadOnlyModelsStore implements ModelsStore {
  constructor(private readonly path: string) {}

  async read(providerId: string): Promise<ModelsStoreEntry | undefined> {
    const file = Bun.file(this.path);
    if (!(await file.exists())) return undefined;
    try {
      const entries = (await file.json()) as Record<string, ModelsStoreEntry>;
      return entries[providerId];
    } catch {
      return undefined; // Mid-write by Pi or corrupt: the built-in catalog still applies.
    }
  }

  async write(): Promise<void> {}

  async delete(): Promise<void> {}
}

/** Create Pi's `ModelRuntime` over `modelsPath` with a read-only catalog store and no network catalog refresh. */
function createPiRuntime(modelsPath: string): Promise<ModelRuntime> {
  return ModelRuntime.create({
    modelsPath,
    modelsStore: new ReadOnlyModelsStore(join(getAgentDir(), "models-store.json")),
    allowModelNetwork: false,
  });
}

/** A file's inode, size and mtime (`missing` while absent): a sync's atomic rename changes the inode. */
async function fileStamp(path: string): Promise<string> {
  try {
    const { ino, size, mtimeMs } = await stat(path);
    return `${ino}:${size}:${mtimeMs}`;
  } catch {
    return "missing";
  }
}

/** Pi's runtime, created on first use (a failed start retries), reloading `models.json` when it changed since the last call. */
export function reloadingPiRuntime(): () => Promise<Models> {
  let runtime: Promise<ModelRuntime> | undefined;
  let loadedStamp: string | undefined;
  let reload: Promise<unknown> | undefined;
  return async () => {
    const modelsPath = join(getAgentDir(), "models.json");
    const stamp = await fileStamp(modelsPath);
    if (!runtime) {
      loadedStamp = stamp; // Stamped before the load, so a change during it reloads on the next call.
      runtime = createPiRuntime(modelsPath).catch((error) => {
        runtime = undefined;
        throw error;
      });
      return runtime;
    }
    const current = await runtime;
    if (stamp !== loadedStamp) {
      loadedStamp = stamp;
      reload = current.refresh({ allowNetwork: false }).catch((error) => {
        loadedStamp = undefined; // Retry on the next call.
        throw error;
      });
    }
    await reload; // Concurrent calls wait for the reload one of them started.
    return current;
  };
}
