/**
 * UI extension points of the Pi transcript (F2, F2b): native React UI for a tool or a `custom` message type.
 *
 * - Tool renderers live in the registry the chat shares (`features/conversation/tools/renderers.ts`), re-exported here
 * - A message renderer turns a `custom` message of its `customType` into a card, or passes by returning `next()`
 *
 * After pi-web.dev's `resolve(toolName, next)` chain.
 */
import type { ReactNode } from "react"
import type { CustomMessage } from "./types"

export { registerToolRenderer, renderToolCall } from "@/features/conversation/tools/renderers"

export type MessageRenderer = (message: CustomMessage, next: () => ReactNode) => ReactNode

const messageRenderers = new Map<string, MessageRenderer>()

/** Render `custom` messages of `customType` with `render`, in place of an earlier registration of that type. */
export function registerMessageRenderer(customType: string, render: MessageRenderer): void {
  messageRenderers.set(customType, render)
}

/** `message` through its `customType`'s renderer, which gets `fallback` (its text) as `next`; `fallback()` without one. */
export function renderCustomMessage(message: CustomMessage, fallback: () => ReactNode): ReactNode {
  const render = messageRenderers.get(message.customType)
  return render ? render(message, fallback) : fallback()
}
