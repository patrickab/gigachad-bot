// Tauri v2 sets TAURI_ENV_* in beforeBuildCommand's environment, so the build
// auto-detects its target: desktop embeds a static export, web keeps a real
// Node server (`npm run start`, SSR/API routes remain possible).
const isTauriBuild = !!process.env.TAURI_ENV_PLATFORM;

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: isTauriBuild ? "export" : "standalone",
  outputFileTracingRoot: __dirname,
  webpack: (config) => {
    config.resolve.alias.canvas = false;
    config.resolve.alias.encoding = false;
    return config;
  },
};

module.exports = nextConfig;
