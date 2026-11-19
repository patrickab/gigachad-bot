/**
 * UI extension points of the Pi transcript (F2, F2b): native React UI for a tool or a `custom` message type.
 *
 * - A tool renderer replaces the default tool row, wraps it (`next()`), or passes per call by returning `next()`
 * - Renderers of one tool chain newest first: `next()` is the one registered before, the last `next()` the default row (`tools/ToolCallRow.tsx`)
 * - `placement: "standalone"` keeps the tool's calls outside the folded "Process details" (todos, plans, questions), the newest registration decides
 * - A message renderer turns a `custom` message of its `customType` into a card, with the same chain
 *
 * After pi-web.dev's `resolve(toolName, next)` chain.
 */
import type { ReactNode } from "react"
import type { CustomMessage, ToolCall, ToolRun } from "./types"

export type ToolRenderer = (call: ToolCall, run: ToolRun | undefined, next: () => ReactNode) => ReactNode
export type MessageRenderer = (message: CustomMessage, next: () => ReactNode) => ReactNode
export type ToolPlacement = "inline" | "standalone"

interface ToolRegistration {
  render: ToolRenderer
  placement: ToolPlacement
}

const toolRenderers = new Map<string, ToolRegistration[]>()
const messageRenderers = new Map<string, MessageRenderer[]>()

/** Add `registration` to `name`'s chain; the returned function removes exactly it. */
function register<T>(registry: Map<string, T[]>, name: string, registration: T): () => void {
  registry.set(name, [...(registry.get(name) ?? []), registration])
  return () => {
    const rest = (registry.get(name) ?? []).filter((existing) => existing !== registration)
    if (rest.length > 0) registry.set(name, rest)
    else registry.delete(name)
  }
}

/** Render `toolName`'s calls with `render`; returns the unregister function. */
export function registerToolRenderer(toolName: string, render: ToolRenderer, options: { placement?: ToolPlacement } = {}): () => void {
  return register(toolRenderers, toolName, { render, placement: options.placement ?? "inline" })
}

/** Render `custom` messages of `customType` with `render`; returns the unregister function. */
export function registerMessageRenderer(customType: string, render: MessageRenderer): () => void {
  return register(messageRenderers, customType, render)
}

/** Where `toolName`'s calls render: inside the folded activity (`inline`) or between the texts (`standalone`). */
export function toolPlacement(toolName: string): ToolPlacement {
  return toolRenderers.get(toolName)?.at(-1)?.placement ?? "inline"
}

/** `call` through its tool's renderer chain, `fallback` (the default row) at the end. */
export function renderToolCall(call: ToolCall, run: ToolRun | undefined, fallback: () => ReactNode): ReactNode {
  const chain = toolRenderers.get(call.name) ?? []
  const resolve = (index: number): ReactNode => (index < 0 ? fallback() : chain[index].render(call, run, () => resolve(index - 1)))
  return resolve(chain.length - 1)
}

/** `message` through its `customType`'s renderer chain, `fallback` (its text) at the end. */
export function renderCustomMessage(message: CustomMessage, fallback: () => ReactNode): ReactNode {
  const chain = messageRenderers.get(message.customType) ?? []
  const resolve = (index: number): ReactNode => (index < 0 ? fallback() : chain[index](message, () => resolve(index - 1)))
  return resolve(chain.length - 1)
}
