/**
 * `GET /usage`: the Anthropic subscription quota, cached process-wide for 60 s.
 *
 * - Consumes: Pi's OAuth login through `Models.getAuth` and `pi-usage-meters`' `tokenFor`/`getJson`,
 *   Anthropic's undocumented `GET /api/oauth/usage`
 * - Provides: `createUsage` -> `(refresh) => Response` with
 *   `{provider:"anthropic", limits:[{window:"5h"|"7d", label?, usedPercent, resetsAt}], fetchedAt}`
 */
import type { Models } from "@earendil-works/pi-ai";
import { getJson, tokenFor } from "pi-usage-meters/extensions/core.js";
import { HttpError } from "./openai.ts";

const USAGE_URL = "https://api.anthropic.com/api/oauth/usage";
const CACHE_MS = 60_000;

export type UsageLimit = { window: "5h" | "7d"; label?: string; usedPercent: number; resetsAt: string | null };
export type UsageReport = { provider: "anthropic"; limits: UsageLimit[]; fetchedAt: string };
export type FetchJson = (url: string, token: string, headers: Record<string, string>) => Promise<unknown>;

const record = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};

function isoOrNull(value: unknown): string | null {
  const ms = typeof value === "string" ? Date.parse(value) : NaN;
  return Number.isFinite(ms) ? new Date(ms).toISOString() : null;
}

function limitOf(window: UsageLimit["window"], percent: unknown, resetsAt: unknown, label?: unknown): UsageLimit | undefined {
  if (typeof percent !== "number" || !Number.isFinite(percent)) return undefined;
  return { window, ...(typeof label === "string" && label ? { label } : {}), usedPercent: percent, resetsAt: isoOrNull(resetsAt) };
}

/** Map Anthropic's usage payload: `limits[]` session -> 5h, weekly -> 7d (model-scoped keeps its model label). */
export function mapAnthropicUsage(payload: unknown): UsageLimit[] {
  const usage = record(payload);
  if (Array.isArray(usage.limits)) {
    return usage.limits.flatMap((entry) => {
      const limit = record(entry);
      const window = limit.group === "session" ? "5h" : limit.group === "weekly" ? "7d" : undefined;
      const label = record(record(record(limit.scope).model)).display_name;
      return (window && limitOf(window, limit.percent, limit.resets_at, label)) || [];
    });
  }
  // Older payloads without limits[]: the fixed five_hour / seven_day windows.
  const fiveHour = record(usage.five_hour);
  const sevenDay = record(usage.seven_day);
  return [
    limitOf("5h", fiveHour.utilization, fiveHour.resets_at),
    limitOf("7d", sevenDay.utilization, sevenDay.resets_at),
  ].filter((limit) => limit !== undefined);
}

/** The status a failed fetch keeps: Anthropic's own (e.g. 429), else 502. */
function failureStatus(error: unknown): number {
  if (error instanceof HttpError) return error.status;
  const status = error && typeof error === "object" && "status" in error ? error.status : undefined;
  return typeof status === "number" && status >= 400 ? status : 502;
}

/** A `/usage` handler with one cache; failures are cached too, so a 429ing endpoint is not hammered. */
export function createUsage(models: () => Promise<Models>, fetchJson: FetchJson = getJson) {
  let cached: { at: number; response: Promise<{ status: number; body: unknown }> } | undefined;

  async function load(): Promise<{ status: number; body: unknown }> {
    try {
      const runtime = await models();
      const token = await tokenFor({ modelRegistry: { getProviderAuth: (id) => runtime.getAuth(id) } }, "anthropic");
      if (!token) throw new HttpError(503, "anthropic is not logged in to Pi with a subscription");
      const payload = await fetchJson(USAGE_URL, token, { "anthropic-beta": "oauth-2025-04-20" });
      const report: UsageReport = { provider: "anthropic", limits: mapAnthropicUsage(payload), fetchedAt: new Date().toISOString() };
      return { status: 200, body: report };
    } catch (error) {
      const status = failureStatus(error);
      const message = error instanceof Error ? error.message : String(error);
      return { status, body: { error: { message, status } } };
    }
  }

  return async function usage(refresh: boolean): Promise<Response> {
    if (refresh || !cached || Date.now() - cached.at > CACHE_MS) cached = { at: Date.now(), response: load() };
    const { status, body } = await cached.response;
    return Response.json(body, { status });
  };
}
