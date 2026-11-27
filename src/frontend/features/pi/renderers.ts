/**
 * UI extension points of the Pi transcript (F2, F2b): native React UI for a tool or a `custom` message type.
 *
 * - Tool renderers and placements live in the registry the chat shares (`features/conversation/tools/renderers.ts`), re-exported here
 * - A message renderer turns a `custom` message of its `customType` into a card; renderers of one type chain newest first, `next()` the one registered before
 *
 * After pi-web.dev's `resolve(toolName, next)` chain.
 */
import type { ReactNode } from "react"
import { register } from "@/features/conversation/tools/renderers"
import type { CustomMessage } from "./types"

export { registerToolRenderer, renderToolCall, toolPlacement } from "@/features/conversation/tools/renderers"

export type MessageRenderer = (message: CustomMessage, next: () => ReactNode) => ReactNode

const messageRenderers = new Map<string, MessageRenderer[]>()

/** Render `custom` messages of `customType` with `render`; returns the unregister function. */
export function registerMessageRenderer(customType: string, render: MessageRenderer): () => void {
  return register(messageRenderers, customType, render)
}

/** `message` through its `customType`'s renderer chain, `fallback` (its text) at the end. */
export function renderCustomMessage(message: CustomMessage, fallback: () => ReactNode): ReactNode {
  const chain = messageRenderers.get(message.customType) ?? []
  const resolve = (index: number): ReactNode => (index < 0 ? fallback() : chain[index](message, () => resolve(index - 1)))
  return resolve(chain.length - 1)
}
