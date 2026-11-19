/**
 * Per-profile Pi arguments (D4, D9, G1): what a child of each profile loads and may use. Host-agnostic plain data, a
 * spawn strategy (local now, sandboxed later) runs the same argv.
 *
 * - Consumes: the profile fields of a `POST /sessions` body
 * - Provides: `ProfileSpec`, `DOCUMENT_TOOLS`, `parseProfileSpec`, `profileArgs`, `sameProfile`
 *
 * `coding` runs Pi with the user's configuration. `document` runs it with the request's system prompt and tool
 * allowlist and nothing discovered: no skills, context files (`AGENTS.md`, `CLAUDE.md`), extensions (built-in MCP
 * included), prompt templates or trust-gated project files (RPC cannot ask for project trust). Pi keeps the system
 * prompt per process, so the client sends it again on every start and resume.
 */
import { PROFILES, type Profile } from "../host/index.ts";

export type ProfileSpec = { profile: "coding" } | { profile: "document"; systemPrompt: string; tools: string[] };

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
  if (profile === "coding") {
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
  if (spec.profile === "coding") return [];
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
  if (a.profile === "coding" || b.profile === "coding") return a.profile === b.profile;
  return a.systemPrompt === b.systemPrompt && a.tools.join(",") === b.tools.join(",");
}
