/**
 * How long a provider likely keeps a session's prompt cached. Providers report no cache lifetime, and Pi sends a long
 * retention only as request parameters, so the lifetimes of a long retention are listed per Pi API.
 *
 * - Consumes: a session's `LastCache` and the retention it was started with (`PrivateState.cacheRetention`)
 * - Provides: `cachedUntil`
 */
import type { LastCache } from "../host/index.ts";

/** A short retention, and any API without a long lifetime below: Anthropic's 5 min, the least providers keep. */
const SHORT_MS = 5 * 60_000;
/** What a long retention asks per Pi API: Anthropic and Bedrock a 1 h cache, OpenAI 30 min to 24 h (the shorter). */
const LONG_MS: Record<string, number> = {
  "anthropic-messages": 60 * 60_000,
  "bedrock-converse-stream": 60 * 60_000,
  "openai-responses": 30 * 60_000,
};

/**
 * Until when the cache of `last` likely stays warm (ISO): its end plus the lifetime.
 *
 * - The latest cache write decides where Pi recorded its lifetime: 1 h for a 1 h write, the short one otherwise
 * - Else `retention` (what the session was started with) picks its API's long lifetime or the short one
 */
export function cachedUntil(last: LastCache, retention: string | undefined): string {
  const long = last.written ? last.written === "long" : retention === "long";
  const lifetime = !long ? SHORT_MS : last.written === "long" ? 60 * 60_000 : LONG_MS[last.api] ?? SHORT_MS;
  return new Date(Date.parse(last.at) + lifetime).toISOString();
}
