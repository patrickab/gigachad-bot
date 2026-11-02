import { copyFileSync } from "node:fs";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const frontendDir = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);

// Tauri v2 sets TAURI_ENV_* in beforeBuildCommand's environment, so the build
// auto-detects its target: desktop embeds a static export, web keeps a real
// Node server (`npm run start`, SSR/API routes remain possible).
const isTauriBuild = !!process.env.TAURI_ENV_PLATFORM;

// Files the browser loads by URL, copied from node_modules into public/ (git-ignored) whenever Next loads this config:
// Vercel and the factory preview start Next directly, an npm pre-script would not run there.
const PUBLIC_RUNTIME_FILES = {
  "pdf.worker.min.mjs": "react-pdf/node_modules/pdfjs-dist/build/pdf.worker.min.mjs",
  "libavoid.wasm": "libavoid-js/dist/libavoid.wasm",
};
for (const [name, source] of Object.entries(PUBLIC_RUNTIME_FILES)) {
  copyFileSync(path.join(frontendDir, "node_modules", source), path.join(frontendDir, "public", name));
}

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: isTauriBuild ? "export" : "standalone",
  outputFileTracingRoot: frontendDir,
  // The notebook editor is client-only (`features/notebook`), these are the build seams `@datalayer/jupyter-react` needs
  // (see its `examples/next-js/next.config.mjs`). `options.webpack` is Next's own webpack: a second copy breaks every module.
  transpilePackages: ["@jupyterlab/settingregistry", "@jupyterlite/settings"],
  webpack: (config, options) => {
    config.resolve.alias.canvas = false;
    config.resolve.alias.encoding = false;
    // JupyterLab packages import json5 by a path webpack cannot resolve and expect a global Buffer.
    config.resolve.alias.json5 = require.resolve("json5/lib/index.js");
    config.resolve.fallback = { ...config.resolve.fallback, buffer: require.resolve("buffer/") };
    config.plugins.push(new options.webpack.ProvidePlugin({ Buffer: ["buffer", "Buffer"] }));
    // The toast and splash styles of JupyterLab's apputils-extension are not used by an embedded notebook, and their `@import '~react-toastify/...'` is a
    // webpack tilde path that Tailwind's own import resolver cannot follow.
    config.plugins.push(new options.webpack.NormalModuleReplacementPlugin(/@jupyterlab[\\/]apputils-extension[\\/]style[\\/]base\.css$/, require.resolve("./features/notebook/empty.css")));
    config.module.rules.push(
      // `@jupyter-widgets/controls` reads its version with `require`, which only a script-mode module may do.
      { test: /@jupyter-widgets[\\/]controls[\\/]lib[\\/]index\.js$/, type: "javascript/auto" },
      { test: /\.js.map$/, type: "asset/resource" },
      // In css, svg is a data URI. In js it is the raw markup string JupyterLab's LabIcon expects.
      { test: /\.svg(\?v=\d+\.\d+\.\d+)?$/, issuer: /\.css$/, use: { loader: "svg-url-loader", options: { encoding: "none", limit: 10000 } } },
      { test: /\.svg(\?v=\d+\.\d+\.\d+)?$/, issuer: /\.js$/, type: "asset/source" },
      { resourceQuery: /text/, type: "asset/resource", generator: { filename: "[name][ext]" } },
      { test: /pypi\/.*/, type: "asset/resource", generator: { filename: "pypi/[name][ext][query]" } },
      { test: /\.whl$/, type: "asset/resource", generator: { filename: "pypi/[name][ext][query]" } },
      { test: /pyodide-kernel-extension\/schema\/.*/, type: "asset/resource", generator: { filename: "schema/[name][ext][query]" } },
    );
    return config;
  },
};

export default nextConfig;
