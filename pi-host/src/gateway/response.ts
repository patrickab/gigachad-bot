/**
 * pi-ai results -> OpenAI chat-completion bodies and SSE chunks.
 *
 * - Consumes: a finished `AssistantMessage` or a pi-ai event stream
 * - Provides: `completionBody` (non-stream) and `sseStream` (chunks, signed `thinking_blocks`, usage chunk, `[DONE]`)
 */
import type { AssistantMessage, AssistantMessageEvent, ThinkingContent, Usage } from "@earendil-works/pi-ai";
import type { OpenAIThinkingBlock } from "./openai.ts";

const finishReason = (reason: AssistantMessage["stopReason"]) =>
  reason === "toolUse" ? "tool_calls" : reason === "length" ? "length" : "stop";

function usageBody(usage: Usage) {
  const prompt = usage.input + usage.cacheRead + usage.cacheWrite;
  return {
    prompt_tokens: prompt,
    completion_tokens: usage.output,
    total_tokens: prompt + usage.output,
    prompt_tokens_details: { cached_tokens: usage.cacheRead, cache_creation_tokens: usage.cacheWrite },
    completion_tokens_details: usage.reasoning === undefined ? undefined : { reasoning_tokens: usage.reasoning },
  };
}

/** A signed (or redacted) thinking block in LiteLLM's shape; unsigned thinking has no block. */
function thinkingBlock(block: ThinkingContent): OpenAIThinkingBlock | undefined {
  if (block.redacted) return { type: "redacted_thinking", data: block.thinkingSignature };
  if (!block.thinkingSignature) return undefined;
  return { type: "thinking", thinking: block.thinking, signature: block.thinkingSignature };
}

/** The non-stream `chat.completion` body for a finished message. */
export function completionBody(message: AssistantMessage, id: string, modelId: string) {
  const text = message.content.flatMap((block) => (block.type === "text" ? [block.text] : [])).join("");
  const reasoning = message.content.flatMap((block) => (block.type === "thinking" ? [block.thinking] : [])).join("");
  const calls = message.content.flatMap((block) =>
    block.type === "toolCall"
      ? [{ id: block.id, type: "function", function: { name: block.name, arguments: JSON.stringify(block.arguments) } }]
      : [],
  );
  const blocks = message.content.flatMap((block) => (block.type === "thinking" ? (thinkingBlock(block) ?? []) : []));
  return {
    id,
    object: "chat.completion",
    created: Math.floor(message.timestamp / 1000),
    model: modelId,
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: text || null,
          ...(calls.length ? { tool_calls: calls } : {}),
          ...(reasoning ? { reasoning_content: reasoning } : {}),
          ...(blocks.length ? { thinking_blocks: blocks } : {}),
        },
        finish_reason: finishReason(message.stopReason),
      },
    ],
    usage: usageBody(message.usage),
  };
}

/** OpenAI SSE for a pi-ai event stream; `onCancel` fires when the client goes away. */
export function sseStream(
  events: AsyncIterable<AssistantMessageEvent>,
  id: string,
  modelId: string,
  includeUsage: boolean,
  onCancel: () => void,
): ReadableStream<Uint8Array> {
  const created = Math.floor(Date.now() / 1000);
  const encoder = new TextEncoder();
  let cancelled = false;
  return new ReadableStream({
    async start(controller) {
      const send = (data: unknown) => {
        if (!cancelled) controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));
      };
      const chunk = (delta: object, finish: string | null = null) =>
        send({ id, object: "chat.completion.chunk", created, model: modelId, choices: [{ index: 0, delta, finish_reason: finish }] });
      // reasoning_content "" keeps LiteLLM from dropping a blocks-only chunk as empty (is_chunk_non_empty).
      const sendThinking = (block: OpenAIThinkingBlock) => chunk({ reasoning_content: "", thinking_blocks: [block] });
      const toolIndex = new Map<number, number>(); // pi contentIndex -> OpenAI tool_calls index
      const sentThinking = new Set<number>(); // contentIndex of thinking blocks already sent with their signature
      try {
        for await (const event of events) {
          switch (event.type) {
            case "start":
              chunk({ role: "assistant", content: "" });
              break;
            case "text_delta":
              chunk({ content: event.delta });
              break;
            case "thinking_delta":
              chunk({ reasoning_content: event.delta });
              break;
            case "thinking_end": {
              const block = event.partial.content[event.contentIndex];
              const signed = block?.type === "thinking" ? thinkingBlock(block) : undefined;
              if (signed) {
                sentThinking.add(event.contentIndex);
                sendThinking(signed);
              }
              break;
            }
            case "toolcall_start": {
              const call = event.partial.content[event.contentIndex];
              if (call?.type !== "toolCall") break;
              const index = toolIndex.size;
              toolIndex.set(event.contentIndex, index);
              chunk({ tool_calls: [{ index, id: call.id, type: "function", function: { name: call.name, arguments: "" } }] });
              break;
            }
            case "toolcall_delta":
              chunk({ tool_calls: [{ index: toolIndex.get(event.contentIndex), function: { arguments: event.delta } }] });
              break;
            case "done":
              // Signatures some providers only settle on the final message.
              event.message.content.forEach((block, index) => {
                const signed = block.type === "thinking" && !sentThinking.has(index) ? thinkingBlock(block) : undefined;
                if (signed) sendThinking(signed);
              });
              chunk({}, finishReason(event.message.stopReason));
              if (includeUsage)
                send({ id, object: "chat.completion.chunk", created, model: modelId, choices: [], usage: usageBody(event.message.usage) });
              break;
            case "error":
              send({ error: { message: event.error.errorMessage ?? event.reason, type: event.reason } });
              break;
          }
        }
      } catch (error) {
        send({ error: { message: String(error), type: "gateway_error" } });
      }
      if (cancelled) return;
      controller.enqueue(encoder.encode("data: [DONE]\n\n"));
      controller.close();
    },
    cancel() {
      cancelled = true;
      onCancel();
    },
  });
}
