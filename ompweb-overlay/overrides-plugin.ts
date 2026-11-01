import { readdirSync } from "node:fs";
import path from "node:path";
import type { BunPlugin } from "bun";

const OVERRIDES_DIR = "custom/overrides";
const MODULE_EXTENSION = /\.(tsx?|jsx?)$/;

/** Upstream module path without extension (e.g. `<root>/components/ChatInput`) -> the file replacing it. */
function listModuleOverrides(projectRoot: string): Map<string, string> {
  const overridesRoot = path.join(projectRoot, OVERRIDES_DIR);
  const files = readdirSync(overridesRoot, { recursive: true, encoding: "utf8" }).filter((file) => MODULE_EXTENSION.test(file));
  return new Map(files.map((file) => [path.join(projectRoot, file.replace(MODULE_EXTENSION, "")), path.join(overridesRoot, file)]));
}

/** The absolute path a `@/` or relative import points at, without extension, or null for a package import. */
function requestedModule(projectRoot: string, request: string, importer: string): string | null {
  const target = request.startsWith("@/") ? path.join(projectRoot, request.slice(2)) : request.startsWith(".") ? path.resolve(path.dirname(importer), request) : null;
  return target && target.replace(MODULE_EXTENSION, "");
}

/**
 * Every file under custom/overrides replaces the upstream module at the same path, whatever the import looked like (`@/x`, `./x`, `../x`).
 * A replacement can wrap the module it replaces: its own imports of it reach the upstream file.
 */
export function customOverrides(projectRoot: string): BunPlugin {
  const overrides = listModuleOverrides(projectRoot);
  return {
    name: "custom-overrides",
    setup(build) {
      build.onResolve({ filter: /^(@\/|\.)/ }, ({ path: request, importer }) => {
        const requested = requestedModule(projectRoot, request, importer);
        const replacement = requested ? overrides.get(requested) : undefined;
        return replacement && replacement !== importer ? { path: replacement } : undefined;
      });
    },
  };
}
