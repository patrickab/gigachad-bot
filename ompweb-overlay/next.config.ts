import { readdirSync } from "fs";
import { basename, dirname, join } from "path";
import type { NextConfig } from "next";
import { withFraming } from "./framing";

type WebpackConfig = Parameters<NonNullable<NextConfig["webpack"]>>[0];

const OVERRIDES_DIR = "custom/overrides";
const MODULE_EXTENSION = /\.(tsx?|jsx?)$/;

/** Upstream module path (e.g. `components/ThemeSwitcher`) and the file replacing it. */
type ModuleOverride = { upstreamModule: string; replacementFile: string };

/** Every file under custom/overrides replaces the upstream module at the same path. */
function listModuleOverrides(projectRoot: string): ModuleOverride[] {
  const overridesRoot = join(projectRoot, OVERRIDES_DIR);
  const files = readdirSync(overridesRoot, { recursive: true, encoding: "utf8" }).filter((file) => MODULE_EXTENSION.test(file));
  return files.map((file) => ({
    upstreamModule: file.replace(MODULE_EXTENSION, "").split("\\").join("/"),
    replacementFile: join(overridesRoot, file),
  }));
}

/**
 * Turbopack aliases requests, not files: cover the `@/` import and a sibling's `./` import.
 * Any other request (e.g. the replacement's `../../../components/X`) reaches the upstream file.
 */
function turbopackAliases(override: ModuleOverride): [string, string][] {
  const target = `./${OVERRIDES_DIR}/${override.upstreamModule}`;
  return [[`@/${override.upstreamModule}`, target], [`./${basename(override.upstreamModule)}`, target]];
}

/** Webpack swaps the resolved upstream file, whatever the import request looked like. */
function upstreamFilePattern(override: ModuleOverride): RegExp {
  const escaped = override.upstreamModule.replace(/[.*+?^${}()|[\]\\]/g, "\\$&").replace(/\//g, "[\\\\/]");
  return new RegExp(`[\\\\/]${escaped}\\.(tsx?|jsx?)$`);
}

/** Webpack's module description after resolution, only the fields the replacement rewrites. */
type CreateData = { resource?: string; context?: string; request?: string; userRequest?: string };
type ResolveData = { contextInfo: { issuer: string }; createData?: CreateData };

/**
 * Swap the resolved upstream file for its replacement, except inside the replacement, so it can wrap upstream.
 *
 * - `context`: the replacement resolves its own relative imports from its own directory
 * - `request`/`userRequest`: the module's identity, else replacement and wrapped upstream would merge into one module
 */
function replaceUnlessImportedByReplacement(override: ModuleOverride) {
  const { replacementFile } = override;
  return ({ contextInfo, createData }: ResolveData) => {
    const upstreamFile = createData?.resource;
    if (!createData || !upstreamFile || contextInfo.issuer === replacementFile) return;
    createData.resource = replacementFile;
    createData.context = dirname(replacementFile);
    createData.request = createData.request?.replace(upstreamFile, replacementFile);
    createData.userRequest = createData.userRequest?.replace(upstreamFile, replacementFile);
  };
}

/**
 * Apply the fork's changes to upstream's Next config.
 *
 * - Module overrides: `next dev` runs Turbopack, `next build --webpack` runs webpack, so both get them
 * - A replacement can wrap the module it replaces: import it by a relative path (`../../../components/X`), never `@/` or `./X`
 * - Framing by configured origins, see `custom/framing.ts`
 */
export function withCustom(upstreamConfig: (phase: string) => NextConfig) {
  return (phase: string): NextConfig => {
    const config = withFraming(upstreamConfig(phase));
    const overrides = listModuleOverrides(config.outputFileTracingRoot ?? process.cwd());
    return {
      ...config,
      turbopack: {
        ...config.turbopack,
        resolveAlias: { ...config.turbopack?.resolveAlias, ...Object.fromEntries(overrides.flatMap(turbopackAliases)) },
      },
      webpack(webpackConfig: WebpackConfig, context) {
        const result = config.webpack ? config.webpack(webpackConfig, context) : webpackConfig;
        for (const override of overrides) {
          result.plugins.push(new context.webpack.NormalModuleReplacementPlugin(upstreamFilePattern(override), replaceUnlessImportedByReplacement(override)));
        }
        return result;
      },
    };
  };
}
