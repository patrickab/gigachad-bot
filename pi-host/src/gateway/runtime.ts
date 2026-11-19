/**
 * Pi's model runtime for the gateway and /usage: Pi's registry, `auth.json` and token refresh.
 *
 * - Consumes: the Pi agent dir (`PI_CODING_AGENT_DIR`, default `~/.pi/agent`): `auth.json`, `models.json`, `models-store.json`
 * - Provides: `createPiRuntime`, a pi-ai `Models` collection shared with running Pi processes
 *
 * Writes: only `auth.json`, when an OAuth token is refreshed (under Pi's own file lock, so a rotated
 * refresh token is never lost to the Pi sessions sharing it). The model catalog store is read-only
 * here and catalogs are never refreshed over the network: Pi itself keeps `models-store.json` current.
 */
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

/** Create Pi's `ModelRuntime` with a read-only catalog store and no network catalog refresh. */
export async function createPiRuntime(): Promise<Models> {
  return ModelRuntime.create({
    modelsStore: new ReadOnlyModelsStore(join(getAgentDir(), "models-store.json")),
    allowModelNetwork: false,
  });
}
