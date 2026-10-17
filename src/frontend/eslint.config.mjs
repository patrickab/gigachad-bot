// Folder boundaries: features own their domain, shared layers never reach into features or app.
import { defineConfig, globalIgnores } from "eslint/config"
import nextVitals from "eslint-config-next/core-web-vitals"
import boundaries from "eslint-plugin-boundaries"
import tseslint from "typescript-eslint"

// A feature's public API: files at its root or at a sub-feature root, never inside concept folders.
const publicEntry = ["*", "*/*"]

const crossFeature = (from, to) => ({
  from: { element: { type: "feature", captured: { feature: from } } },
  allow: { to: { element: { type: "feature", captured: { feature: to }, fileInternalPath: publicEntry } } },
})

// Next's plugins are registered without their rules so existing `eslint-disable` comments for them still resolve.
const nextPlugins = Object.assign({}, ...nextVitals.map((config) => config.plugins ?? {}))

export default defineConfig([
  globalIgnores([".next/**", "out/**", "build/**", "public/**", "next-env.d.ts"]),
  {
    files: ["{app,components,contexts,features,hooks,lib}/**/*.{ts,tsx}"],
    languageOptions: { parser: tseslint.parser },
    linterOptions: { reportUnusedDisableDirectives: "off" },
    plugins: { ...nextPlugins, boundaries },
    settings: {
      "import/resolver": { typescript: { alwaysTryTypes: true } },
      "boundaries/elements": [
        { type: "app", pattern: "app", partialMatch: false },
        { type: "feature", pattern: "features/*", capture: ["feature"], partialMatch: false },
        { type: "shared", pattern: "{components,contexts,hooks,lib}", partialMatch: false },
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
          { from: { element: { type: "app" } }, allow: { to: { element: { type: "feature", fileInternalPath: publicEntry } } } },
          crossFeature("chat", "editor"),
          crossFeature("navigation", "memory"),
        ],
      }],
    },
  },
])
