/**
 * Per-profile Pi arguments (D4, D9, G1, C5): what a child of each profile loads and may use. Host-agnostic plain data,
 * each spawn strategy (the Pi binary, or `agent-sandbox pi-rpc` for `sandboxed`) runs the same argv.
 *
 * - Consumes: the profile fields of a `POST /sessions` body
 * - Provides: `ProfileSpec`, `DOCUMENT_TOOLS`, `parseProfileSpec`, `profileArgs`, `sameProfile`, `childModel`
 *
 * `coding` runs Pi with the user's configuration. `document` runs it with the request's system prompt and tool
 * allowlist and nothing discovered: no skills, context files (`AGENTS.md`, `CLAUDE.md`), extensions (built-in MCP
 * included), prompt templates or trust-gated project files (RPC cannot ask for project trust). Pi keeps the system
 * prompt per process, so the client sends it again on every start and resume. `sandboxed` takes `coding`'s arguments;
 * its container knows models only as the `gateway` provider (C2).
 */
import { PROFILES, type Profile } from "../host/index.ts";

export type ProfileSpec =
  | { profile: "coding" | "sandboxed" }
  | { profile: "document"; systemPrompt: string; tools: string[] };

export type ModelRef = { provider: string; id: string };

/** G1: the workbench prompts edit and find files, never run commands. */
export const DOCUMENT_TOOLS: readonly string[] = ["read", "edit", "write", "grep", "find", "ls"];

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
  if (tools === undefined) return { profile: "document", systemPrompt, tools: [...DOCUMENT_TOOLS] };
  if (!Array.isArray(tools) || tools.length === 0 || !tools.every((tool) => typeof tool === "string" && TOOL_NAME.test(tool))) {
    return { error: "invalid_tools" };
  }
  return { profile: "document", systemPrompt, tools };
}

/** Pi arguments a profile adds after `--mode rpc` (Pi 1.5.2 `cli.md`). */
export function profileArgs(spec: ProfileSpec): string[] {
  if (spec.profile !== "document") return [];
  return [
    "--tools",
    spec.tools.join(","),
    "--system-prompt",
    spec.systemPrompt,
    "--no-skills",
    "--no-context-files",
    "--no-extensions",
    "--no-prompt-templates",
    "--no-approve",
  ];
}

/** True when a child started for `a` serves a request for `b`. */
export function sameProfile(a: ProfileSpec, b: ProfileSpec): boolean {
  if (a.profile !== "document" || b.profile !== "document") return a.profile === b.profile;
  return a.systemPrompt === b.systemPrompt && a.tools.join(",") === b.tools.join(",");
}

/** The model a child of `spec` selects for a requested `<provider>/<id>`: sandboxed children reach it as `gateway` (C2). */
export function childModel(spec: ProfileSpec, model: ModelRef): ModelRef {
  return spec.profile === "sandboxed" ? { provider: "gateway", id: `${model.provider}/${model.id}` } : model;
}
