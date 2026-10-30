import { existsSync, readdirSync } from "node:fs";
import { basename, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { NextConfig } from "next";
import { frameAncestorOrigins, withFraming } from "./framing";

/** The upstream worktree this file lands in as custom/next.config.ts, wherever the build runs from. */
export const PROJECT_ROOT = fileURLToPath(new URL("..", import.meta.url));

const OVERRIDES_DIR = "custom/overrides";
const MODULE_EXTENSION = /\.(tsx?|jsx?)$/;

/**
 * Upstream module paths (e.g. `components/model-picker`) that a file under custom/overrides replaces.
 * The directory is optional: without it nothing is replaced.
 */
export function listOverriddenModules(projectRoot: string): string[] {
  const overridesRoot = join(projectRoot, OVERRIDES_DIR);
  if (!existsSync(overridesRoot)) return [];
  return readdirSync(overridesRoot, { recursive: true, encoding: "utf8" })
    .filter((file) => MODULE_EXTENSION.test(file))
    .map((file) => file.replace(MODULE_EXTENSION, "").split("\\").join("/"));
}

/**
 * Turbopack aliases requests, not files: cover the `@/` import, also with the `.tsx`/`.ts` suffix LearnHarness writes on
 * many imports (`@/components/settings/settings-dialog.tsx`), and a sibling's bare `./X` import.
 *
 * - Never a suffixed `./X.tsx`/`./X.ts` key: Turbopack lets those capture `./<dir>/X` too, so an overlay module
 *   `./embed/X` (and `@/custom/embed/X`) would resolve to the override of `X`. No upstream sibling import of an
 *   overridden module carries a suffix
 * - Any other request (e.g. the replacement's `../../../components/X`) reaches the upstream file, so a replacement can wrap it
 */
export function overrideAliases(upstreamModules: string[]): Record<string, string> {
  return Object.fromEntries(
    upstreamModules.flatMap((upstreamModule): [string, string][] => {
      const target = `./${OVERRIDES_DIR}/${upstreamModule}`;
      const absolute = ["", ".tsx", ".ts"].map((suffix): [string, string] => [`@/${upstreamModule}${suffix}`, target]);
      return [...absolute, [`./${basename(upstreamModule)}`, target]];
    }),
  );
}

/**
 * Apply the overlay to upstream's Next config.
 *
 * - Every file under `custom/overrides` replaces the upstream module at the same path
 * - Only `GIGACHAD_LEARNHARNESS_FRAME_ANCESTORS` may frame the app and talk to its bridge (`./framing`)
 * - Object in, object out: upstream's own tests read the default export of `next.config.ts` as a `NextConfig`
 * - Turbopack only, LearnHarness builds and serves with it (`next build`, `next dev`)
 * - A replacement wraps upstream by importing it by a relative path (`../../../components/X`), never `@/` or `./X`
 */
export function withCustom(upstreamConfig: NextConfig, projectRoot = PROJECT_ROOT, origins = frameAncestorOrigins()): NextConfig {
  const aliases = overrideAliases(listOverriddenModules(projectRoot));
  return withFraming(
    {
      ...upstreamConfig,
      turbopack: {
        ...upstreamConfig.turbopack,
        resolveAlias: { ...upstreamConfig.turbopack?.resolveAlias, ...aliases },
      },
    },
    origins,
  );
}
