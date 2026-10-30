import type { NextConfig } from "next";

/** Shell origins allowed to frame LearnHarness and drive its bridge, space or comma separated (run-learnharness.sh sets it). */
export const FRAME_ANCESTORS_ENV = "GIGACHAD_LEARNHARNESS_FRAME_ANCESTORS";
const ORIGIN_PATTERN = /^https?:\/\/[a-z0-9.-]+(:\d+)?$/i;

/** Origins from `GIGACHAD_LEARNHARNESS_FRAME_ANCESTORS`, an invalid entry throws (the build fails naming it). */
export function frameAncestorOrigins(rawValue = process.env[FRAME_ANCESTORS_ENV] ?? ""): string[] {
  const origins = rawValue.split(/[\s,]+/).filter(Boolean);
  const invalidOrigin = origins.find((origin) => !ORIGIN_PATTERN.test(origin));
  if (invalidOrigin) throw new Error(`${FRAME_ANCESTORS_ENV}: "${invalidOrigin}" is not an origin like http://127.0.0.1:2999`);
  return origins;
}

/**
 * Only the shell's origins may frame LearnHarness, and only they reach its bridge.
 *
 * - CSP `frame-ancestors 'self' <origins>` on every page (upstream sends none), 'self' keeps LearnHarness's own frames
 * - The origins reach the browser bundle as `process.env.GIGACHAD_LEARNHARNESS_FRAME_ANCESTORS` (`embed/bridge.tsx`)
 * - Both are fixed at build time, the launcher rebuilds when the list changes
 */
export function withFraming(config: NextConfig, origins: string[]): NextConfig {
  const upstreamHeaders = config.headers;
  const frameAncestors = ["'self'", ...origins].join(" ");
  return {
    ...config,
    env: { ...config.env, [FRAME_ANCESTORS_ENV]: origins.join(" ") },
    async headers() {
      return [
        ...((await upstreamHeaders?.()) ?? []),
        { source: "/:path*", headers: [{ key: "Content-Security-Policy", value: `frame-ancestors ${frameAncestors}` }] },
      ];
    },
  };
}
