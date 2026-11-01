const FRAME_ANCESTORS_ENV = "OMP_WEB_FRAME_ANCESTORS";
const ORIGIN_PATTERN = /^https?:\/\/[a-z0-9.-]+(:\d+)?$/i;

/** Origins from `OMP_WEB_FRAME_ANCESTORS` (space or comma separated), an invalid entry throws. */
export function frameAncestorOrigins(rawValue = process.env[FRAME_ANCESTORS_ENV] ?? ""): string[] {
  const origins = rawValue.split(/[\s,]+/).filter(Boolean);
  const invalidOrigin = origins.find((origin) => !ORIGIN_PATTERN.test(origin));
  if (invalidOrigin) throw new Error(`${FRAME_ANCESTORS_ENV}: "${invalidOrigin}" is not an origin like http://127.0.0.1:2999`);
  return origins;
}

// Read once at server start, so an invalid value stops the server instead of failing every response.
const configuredOrigins = frameAncestorOrigins();

/**
 * Let the origins in `OMP_WEB_FRAME_ANCESTORS` embed the app in a frame (gigachad-bot's shell).
 *
 * - Unset or empty: upstream's headers unchanged (`DENY`, `frame-ancestors 'none'`)
 * - Set: CSP `frame-ancestors` lists them and `X-Frame-Options: DENY` goes
 */
export function framedHeaders(headers: Record<string, string>, origins = configuredOrigins): Record<string, string> {
  if (origins.length === 0) return headers;
  const framed = { ...headers };
  delete framed["X-Frame-Options"];
  const policy = framed["Content-Security-Policy"];
  if (policy) framed["Content-Security-Policy"] = policy.replace("frame-ancestors 'none'", `frame-ancestors 'self' ${origins.join(" ")}`);
  return framed;
}
