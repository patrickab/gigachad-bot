/** Catalog entry points bound to a `LocalHost`, the shapes the catalog tests call. */
import * as catalog from "../../src/catalog/index.ts";
import { LocalHost } from "../../src/host/local/index.ts";
import type { LiveSessions } from "../../src/supervisor/index.ts";

const host = new LocalHost({ piBin: "pi", extraChildEnv: [] });

export function handleCatalog(request: Request, url: URL, live: LiveSessions): Promise<Response | undefined> {
  return catalog.handleCatalog(request, url, live, host);
}

export function isPinned(sessionPath: string): Promise<boolean> {
  return catalog.isPinned(host, sessionPath);
}

export function copyPrivateState(fromPath: string, toPath: string): Promise<void> {
  return catalog.copyPrivateState(host, fromPath, toPath);
}
