/**
 * OpenAI chat-completions wire types the gateway accepts and the HTTP error it maps to a status.
 *
 * - Consumes: nothing
 * - Provides: request/message shapes (LiteLLM's `thinking_blocks` convention included), `HttpError`, `errorResponse`
 */
import type { ModelThinkingLevel } from "@earendil-works/pi-ai";

export type OpenAIPart = { type: "text"; text: string } | { type: "image_url"; image_url: { url: string } | string };

export type OpenAIToolCall = { id: string; type: "function"; function: { name: string; arguments: string } };

/** LiteLLM's Anthropic convention: signed thinking carried across turns. */
export type OpenAIThinkingBlock =
  | { type: "thinking"; thinking: string; signature?: string }
  | { type: "redacted_thinking"; data?: string };

export type OpenAIMessage = {
  role: "system" | "developer" | "user" | "assistant" | "tool";
  content?: string | OpenAIPart[] | null;
  tool_calls?: OpenAIToolCall[];
  tool_call_id?: string;
  thinking_blocks?: OpenAIThinkingBlock[];
};

export type OpenAITool = {
  type: "function";
  function: { name: string; description?: string; parameters?: Record<string, unknown>; strict?: boolean };
};

export type OpenAIRequest = {
  model: string;
  messages: OpenAIMessage[];
  tools?: OpenAITool[];
  tool_choice?: "auto" | "none" | "required" | object;
  stream?: boolean;
  stream_options?: { include_usage?: boolean };
  max_tokens?: number;
  max_completion_tokens?: number;
  temperature?: number;
  reasoning_effort?: "none" | ModelThinkingLevel;
};

/** An error that leaves the gateway with its own HTTP status. */
export class HttpError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

/** OpenAI-style error body; `HttpError` keeps its status, anything else is a 500. */
export function errorResponse(error: unknown): Response {
  const status = error instanceof HttpError ? error.status : 500;
  const message = error instanceof Error ? error.message : String(error);
  return Response.json({ error: { message, type: "gateway_error", code: status } }, { status });
}
