/** A tool call and its run as a tool row reads them: Pi's protocol shapes (`features/pi/types.ts` re-exports them), which the chat adapts its calls to. */

export interface TextContent { type: "text"; text: string }
export interface ImageContent { type: "image"; data: string; mimeType: string }
/** A tool-result image as pi-host sends it: stored under the sha256 of its bytes, fetched from `/api/pi/media/<hash>` (`piMediaUrl`). */
export interface MediaImage { type: "image"; mimeType: string; hash: string }
/** An image block of a tool result: a stored one, or inline base64 when pi-host could not store it. */
export type ToolImage = ImageContent | MediaImage

export interface ToolCall { type: "toolCall"; id: string; name: string; arguments: Record<string, unknown> }

/** Live state of one tool call, keyed by `toolCallId`, from `tool_execution_*` events. */
export interface ToolRun {
  /** Latest partial result while running, the final result once ended. */
  result?: { content: (TextContent | ToolImage)[]; details?: unknown }
  isError: boolean
  running: boolean
  durationMs?: number
}
