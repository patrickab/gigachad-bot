import { copyFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const frontendDir = path.dirname(fileURLToPath(import.meta.url));

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
  webpack: (config) => {
    config.resolve.alias.canvas = false;
    config.resolve.alias.encoding = false;
    return config;
  },
};

export default nextConfig;
