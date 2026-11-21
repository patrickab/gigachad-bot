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
  // A factory preview is served under a path of the preview gate (factory.yaml sets it), `appPath` reads the same variable.
  basePath: process.env.NEXT_PUBLIC_BASE_PATH || undefined,
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
    // The notebook draws Plotly output with the prebuilt Plotly the chat's plots use, one file. Its own dependency is an older
    // Plotly as some 1500 source modules, which a build bundled as a second copy.
    config.resolve.alias["plotly.js$"] = require.resolve("plotly.js-dist-min");
    // `next dev` renders the app in the browser only (`app/_components/AppEntry.dev.tsx`), and its server bundle gets a stub
    // of the app: webpack builds what a browser-only import reaches for the server too, here some 11000 modules a second time.
    if (options.dev) {
      const appFile = (name) => path.join(frontendDir, "app/_components", name);
      config.plugins.push(new options.webpack.NormalModuleReplacementPlugin(/app[\\/]_components[\\/]AppEntry\.tsx$/, appFile("AppEntry.dev.tsx")));
      if (options.isServer) config.plugins.push(new options.webpack.NormalModuleReplacementPlugin(/app[\\/]_components[\\/]App\.tsx$/, appFile("App.dev-server.tsx")));
      // The notebook editor is a third of the modules (JupyterLab, Primer, their icon sets): `next dev` compiles it when the first
      // notebook opens, not at startup. webpack tells the page through its own port on localhost, so a browser on another machine
      // never gets the editor: GIGACHAD_EAGER_NOTEBOOK=1 compiles it up front as a build does.
      if (!options.isServer && !process.env.GIGACHAD_EAGER_NOTEBOOK) {
        config.experiments = {
          ...config.experiments,
          lazyCompilation: { entries: false, imports: true, test: (module) => /features[\\/]notebook[\\/]NotebookEditor\.tsx$/.test(module.nameForCondition() ?? "") },
        };
      }
    }
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
