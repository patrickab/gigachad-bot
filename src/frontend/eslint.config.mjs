// Next's recommended React, hooks, a11y and TypeScript rules, plus folder boundaries: features own their domain, shared layers never reach into features or app.
import { join } from "node:path"
import { defineConfig, globalIgnores } from "eslint/config"
import nextVitals from "eslint-config-next/core-web-vitals"
import nextTypescript from "eslint-config-next/typescript"
import boundaries from "eslint-plugin-boundaries"

// A feature's public API: files at its root or at a sub-feature root, never inside concept folders.
const publicEntry = ["*", "*/*"]

// Package files any module may import: all but the omp protocol (`packages/chat-ui/src/protocol.ts`), which only `features/omp` may.
const PACKAGE_FILES_BESIDES_PROTOCOL = "src/!(protocol).*"

const crossFeature = (from, to) => ({
  from: { element: { type: "feature", captured: { feature: from } } },
  allow: { to: { element: { type: "feature", captured: { feature: to }, fileInternalPath: publicEntry } } },
})


export default defineConfig([
  ...nextVitals,
  ...nextTypescript,
  globalIgnores([".next/**", "out/**", "build/**", "public/**", "next-env.d.ts"]),
  {
    files: ["{app,components,contexts,features,hooks,lib}/**/*.{ts,tsx}"],
    plugins: { boundaries },
    settings: {
      "import/resolver": { typescript: { alwaysTryTypes: true } },
      "boundaries/elements": [
        { type: "app", pattern: "app", partialMatch: false },
        { type: "feature", pattern: "features/*", capture: ["feature"], partialMatch: false },
        { type: "shared", pattern: "{components,contexts,hooks,lib}", partialMatch: false },
        { type: "package", pattern: "packages/*", partialMatch: false },
      ],
    },
    rules: {
      "boundaries/dependencies": [2, {
        default: "disallow",
        message: "{{from.element.type}} {{from.element.captured.feature}} may not import {{to.element.path}}/{{to.element.fileInternalPath}}",
        policies: [
          { allow: { to: { module: { origin: ["external", "core"] } } } },
          { allow: { dependency: { relationship: { to: "internal" } } } },
          { from: { element: { type: ["app", "feature", "shared"] } }, allow: { to: { element: { type: "shared" } } } },
          // The omp protocol is the one package file the shell keeps behind its omp adapter.
          { from: { element: { type: ["app", "feature", "shared"] } }, allow: { to: { element: { type: "package", fileInternalPath: PACKAGE_FILES_BESIDES_PROTOCOL } } } },
          { from: { element: { type: "app" } }, allow: { to: { element: { type: "feature", fileInternalPath: publicEntry } } } },
          { from: { element: { type: "feature", captured: { feature: "omp" } } }, allow: { to: { element: { type: "package" } } } },
          crossFeature("chat", "editor"),
          crossFeature("factory", "editor"),
          crossFeature("navigation", "factory"),
          crossFeature("navigation", "memory"),
          crossFeature("navigation", "omp"),
          crossFeature("omp", "chat"),
          crossFeature("omp", "editor"),
          crossFeature("omp", "settings"),
        ],
      }],
    },
  },
  {
    // packages/* ship as tarballs to other apps, so they cannot reach this app's `@/` modules.
    files: ["packages/**/*.{ts,tsx}"],
    ignores: ["packages/**/__tests__/**"],
    rules: {
      "no-restricted-imports": ["error", { patterns: [{ group: ["@/*"], message: "A package must not import the app it ships from." }] }],
      // Only what the package's own manifest declares ships to a consumer, an import that resolves through the shell's node_modules breaks in the fork.
      "import/no-extraneous-dependencies": ["error", { packageDir: [join(import.meta.dirname, "packages/chat-ui")], includeTypes: true }],
    },
  },
])
