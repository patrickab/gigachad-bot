import type { NextConfig } from "next";
import type { Header as HeaderRule } from "next/dist/lib/load-custom-routes";

const FRAME_ANCESTORS_ENV = "OMP_WEB_FRAME_ANCESTORS";
const ORIGIN_PATTERN = /^https?:\/\/[a-z0-9.-]+(:\d+)?$/i;

/** Origins from `OMP_WEB_FRAME_ANCESTORS` (space or comma separated), an invalid entry throws. */
export function frameAncestorOrigins(rawValue = process.env[FRAME_ANCESTORS_ENV] ?? ""): string[] {
  const origins = rawValue.split(/[\s,]+/).filter(Boolean);
  const invalidOrigin = origins.find((origin) => !ORIGIN_PATTERN.test(origin));
  if (invalidOrigin) throw new Error(`${FRAME_ANCESTORS_ENV}: "${invalidOrigin}" is not an origin like http://127.0.0.1:2999`);
  return origins;
}

/** Let `origins` frame a rule's pages: CSP `frame-ancestors` lists them, `X-Frame-Options: DENY` goes. */
function allowFramingBy(origins: string[], rule: HeaderRule): HeaderRule {
  const headers = rule.headers
    .filter((header) => !(header.key === "X-Frame-Options" && header.value === "DENY"))
    .map((header) => header.key === "Content-Security-Policy"
      ? { ...header, value: header.value.replace("frame-ancestors 'none'", `frame-ancestors 'self' ${origins.join(" ")}`) }
      : header);
  return { ...rule, headers };
}

/**
 * Let the origins in `OMP_WEB_FRAME_ANCESTORS` embed the app in a frame (gigachad-bot's shell).
 *
 * - Unset or empty: upstream's headers unchanged (`DENY`, `frame-ancestors 'none'`)
 * - Only rules carrying `DENY` / `frame-ancestors 'none'` change, `/api/files` keeps `SAMEORIGIN`
 */
export function withFraming(config: NextConfig, origins = frameAncestorOrigins()): NextConfig {
  const upstreamHeaders = config.headers;
  if (!upstreamHeaders || origins.length === 0) return config;
  return {
    ...config,
    async headers() {
      return (await upstreamHeaders()).map((rule) => allowFramingBy(origins, rule));
    },
  };
}
