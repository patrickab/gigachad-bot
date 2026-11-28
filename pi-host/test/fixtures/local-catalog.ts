/** Catalog entry points bound to a `LocalHost`, the shapes the catalog tests call. */
import * as catalog from "../../src/catalog/index.ts";
import type { Profile } from "../../src/host/index.ts";
import { LocalHost } from "../../src/host/local/index.ts";
import * as sidecar from "../../src/sidecar.ts";
import { loadConfig } from "../../src/supervisor/config.ts";
import type { LiveSessions } from "../../src/supervisor/index.ts";

const host = new LocalHost(loadConfig());

export function handleCatalog(request: Request, url: URL, live: LiveSessions): Promise<Response | undefined> {
  return catalog.handleCatalog(request, url, live, host);
}

export function isPinned(sessionPath: string): Promise<boolean> {
  return sidecar.isPinned(host, sessionPath);
}

export function copyPrivateState(fromPath: string, toPath: string): Promise<void> {
  return sidecar.copyPrivateState(host, fromPath, toPath);
}

export function markSpawn(sessionPath: string, profile: Profile, retention: string | undefined): Promise<void> {
  return sidecar.markSpawn(host, sessionPath, profile, retention);
}
