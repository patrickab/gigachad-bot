/**
 * Per-profile Pi arguments (D4, D9, G1, C5): what a child of each profile loads and may use. Host-agnostic plain data,
 * each spawn strategy (the Pi binary, or `agent-sandbox pi-rpc` for `sandboxed`) runs the same argv.
 *
 * - Consumes: the profile fields of a `POST /sessions` body
 * - Provides: `ProfileSpec`, `parseProfileSpec`, `profileArgs`, `sameProfile`, `childModels`
 *
 * `coding` runs Pi with the user's configuration. `document` runs it the same way with the request's system prompt
 * appended. A `document` request with a tool allowlist restricts it to those tools and nothing discovered: no skills,
 * extensions (built-in MCP included), prompt templates or trust-gated project files (RPC cannot ask for project trust).
 * Pi keeps the system prompt per process, so the client sends it again on every start and resume. `sandboxed` takes
 * `coding`'s arguments; its container knows models only as the `gateway` provider (C2). Local children have their own
 * providers plus `gateway` for the rest (`extensions/gateway-provider.ts`).
 */
import { PROFILES, type Profile } from "../host/index.ts";

export type ProfileSpec =
  | { profile: "coding" | "sandboxed" }
  | { profile: "document"; systemPrompt: string; tools?: string[] };

export type ModelRef = { provider: string; id: string };

/** Linux caps one argv string at 128 KiB including its NUL (MAX_ARG_STRLEN). */
const MAX_SYSTEM_PROMPT_BYTES = 128 * 1024 - 1;
/** A plain tool name or `*` pattern: `+name`/`-name` would turn `--tools` into an edit of the defaults. */
const TOOL_NAME = /^[\w*][\w*.-]*$/;

/** The profile fields of a `POST /sessions` body as a spec, or the 400 error code. */
export function parseProfileSpec(fields: { profile: unknown; systemPrompt: unknown; tools: unknown }): ProfileSpec | { error: string } {
  const { profile, systemPrompt, tools } = fields;
  if (!PROFILES.includes(profile as Profile)) return { error: "invalid_profile" };
  if (profile === "coding" || profile === "sandboxed") {
    if (systemPrompt !== undefined) return { error: "invalid_system_prompt" };
    if (tools !== undefined) return { error: "invalid_tools" };
    return { profile };
  }
  if (
    typeof systemPrompt !== "string" ||
    systemPrompt.trim() === "" ||
    Buffer.byteLength(systemPrompt) > MAX_SYSTEM_PROMPT_BYTES
  ) {
    return { error: "invalid_system_prompt" };
  }
  if (tools === undefined) return { profile: "document", systemPrompt };
  if (!Array.isArray(tools) || tools.length === 0 || !tools.every((tool) => typeof tool === "string" && TOOL_NAME.test(tool))) {
    return { error: "invalid_tools" };
  }
  return { profile: "document", systemPrompt, tools };
}

/** Pi arguments a profile adds after `--mode rpc` (Pi 1.5.2 `cli.md`). */
export function profileArgs(spec: ProfileSpec): string[] {
  if (spec.profile !== "document") return [];
  const prompt = ["--append-system-prompt", spec.systemPrompt];
  if (!spec.tools) return prompt;
  return ["--tools", spec.tools.join(","), ...prompt, "--no-skills", "--no-extensions", "--no-prompt-templates", "--no-approve"];
}

/** True when a child started for `a` serves a request for `b`. */
export function sameProfile(a: ProfileSpec, b: ProfileSpec): boolean {
  if (a.profile !== "document" || b.profile !== "document") return a.profile === b.profile;
  return a.systemPrompt === b.systemPrompt && a.tools?.join(",") === b.tools?.join(",");
}

/**
 * The models a child of `spec` tries, in order, for a requested `<provider>/<id>`: sandboxed children reach it only as
 * `gateway` (C2), local ones by its own provider when they have it, else through `gateway`.
 */
export function childModels(spec: ProfileSpec, model: ModelRef): ModelRef[] {
  const viaGateway = { provider: "gateway", id: `${model.provider}/${model.id}` };
  if (model.provider === "gateway") return [model];
  return spec.profile === "sandboxed" ? [viaGateway] : [model, viaGateway];
}
