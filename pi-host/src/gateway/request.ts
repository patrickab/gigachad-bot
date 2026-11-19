/**
 * OpenAI request -> pi-ai context and stream options.
 *
 * - Consumes: an `OpenAIRequest` and the resolved pi-ai model
 * - Provides: `toContext` (messages, images, tools, signed thinking back in) and `streamOptions`
 *   (tool_choice, clamped reasoning_effort, max tokens, temperature)
 */
import {
  type Api,
  type AssistantMessage,
  clampThinkingLevel,
  type Context,
  type ImageContent,
  type Message,
  type JsonObject,
  type Model,
  type ModelsSimpleStreamOptions,
  type TextContent,
  type ThinkingLevel,
  type Tool,
  type Usage,
} from "@earendil-works/pi-ai";
import { HttpError, type OpenAIMessage, type OpenAIRequest } from "./openai.ts";

async function imagePart(url: string, signal: AbortSignal): Promise<ImageContent> {
  const data = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (data) return { type: "image", mimeType: data[1]!, data: data[2]! };
  const response = await fetch(url, { signal });
  if (!response.ok) throw new HttpError(400, `image fetch ${response.status}: ${url}`);
  return {
    type: "image",
    mimeType: response.headers.get("content-type")?.split(";")[0] ?? "image/png",
    data: Buffer.from(await response.arrayBuffer()).toString("base64"),
  };
}

async function parts(content: OpenAIMessage["content"], signal: AbortSignal): Promise<(TextContent | ImageContent)[]> {
  if (content == null) return [];
  if (typeof content === "string") return [{ type: "text", text: content }];
  return Promise.all(
    content.map((part) =>
      part.type === "text"
        ? { type: "text" as const, text: part.text }
        : imagePart(typeof part.image_url === "string" ? part.image_url : part.image_url.url, signal),
    ),
  );
}

const textOf = (content: OpenAIMessage["content"]) =>
  typeof content === "string"
    ? content
    : (content ?? []).map((part) => (part.type === "text" ? part.text : "")).join("");

const zeroUsage = (): Usage => ({
  input: 0,
  output: 0,
  cacheRead: 0,
  cacheWrite: 0,
  totalTokens: 0,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
});

function toolArguments(raw: string): JsonObject {
  if (!raw) return {};
  try {
    return JSON.parse(raw);
  } catch {
    throw new HttpError(400, `tool call arguments are not JSON: ${raw.slice(0, 200)}`);
  }
}

function assistantMessage(message: OpenAIMessage, model: Model<Api>, timestamp: number): AssistantMessage {
  const content: AssistantMessage["content"] = [];
  for (const block of message.thinking_blocks ?? []) {
    if (block.type === "thinking")
      content.push({ type: "thinking", thinking: block.thinking ?? "", thinkingSignature: block.signature });
    else content.push({ type: "thinking", thinking: "", thinkingSignature: block.data, redacted: true });
  }
  const text = textOf(message.content);
  if (text) content.push({ type: "text", text });
  for (const call of message.tool_calls ?? []) {
    content.push({
      type: "toolCall",
      id: call.id,
      name: call.function.name,
      arguments: toolArguments(call.function.arguments),
    });
  }
  // Marked as the target model so pi-ai keeps signed thinking blocks (it drops them for other models).
  return {
    role: "assistant",
    content,
    api: model.api,
    provider: model.provider,
    model: model.id,
    usage: zeroUsage(),
    stopReason: message.tool_calls?.length ? "toolUse" : "stop",
    timestamp,
  };
}

/** Translate the OpenAI messages and tools into a pi-ai context for `model`. */
export async function toContext(body: OpenAIRequest, model: Model<Api>, signal: AbortSignal): Promise<Context> {
  if (!Array.isArray(body.messages)) throw new HttpError(400, "messages must be an array");
  const system: string[] = [];
  const messages: Message[] = [];
  const toolNames = new Map<string, string>();
  const now = Date.now();
  for (const message of body.messages) {
    switch (message.role) {
      case "system":
      case "developer":
        // Leading system turns form the prompt, later ones stay in place as pi system messages.
        if (messages.length === 0) system.push(textOf(message.content));
        else messages.push({ role: "system", content: textOf(message.content), timestamp: now });
        break;
      case "user":
        messages.push({ role: "user", content: await parts(message.content, signal), timestamp: now });
        break;
      case "assistant":
        for (const call of message.tool_calls ?? []) toolNames.set(call.id, call.function.name);
        messages.push(assistantMessage(message, model, now));
        break;
      case "tool":
        messages.push({
          role: "toolResult",
          toolCallId: message.tool_call_id ?? "",
          toolName: toolNames.get(message.tool_call_id ?? "") ?? "unknown",
          content: await parts(message.content, signal),
          isError: false,
          timestamp: now,
        });
        break;
      default:
        throw new HttpError(400, `unsupported message ${JSON.stringify(message).slice(0, 200)}`);
    }
  }
  // OpenAI JSON Schema passes through unchanged: pi-ai tools take plain JSON Schema.
  const tools: Tool[] | undefined = body.tools?.map(({ function: definition }) => ({
    name: definition.name,
    description: definition.description ?? "",
    parameters: definition.parameters ?? { type: "object", properties: {} },
    ...(definition.strict ? { constrainedSampling: { type: "json_schema" as const, strict: "prefer" as const } } : {}),
  }));
  return { systemPrompt: system.length ? system.join("\n\n") : undefined, messages, tools };
}

/** Stream options: tool_choice degrades to auto/none, effort is clamped, temperature drops while thinking. */
export function streamOptions(body: OpenAIRequest, model: Model<Api>, signal: AbortSignal): ModelsSimpleStreamOptions {
  const effort = body.reasoning_effort;
  const clamped = effort && effort !== "none" && model.reasoning ? clampThinkingLevel(model, effort) : undefined;
  const reasoning = clamped && clamped !== "off" ? (clamped as ThinkingLevel) : undefined;
  const toolChoice = body.tool_choice === undefined ? undefined : body.tool_choice === "none" ? "none" : "auto";
  return {
    signal,
    reasoning,
    toolChoice,
    maxTokens: body.max_completion_tokens ?? body.max_tokens,
    // Anthropic rejects temperature together with thinking.
    temperature: reasoning ? undefined : body.temperature,
  };
}
