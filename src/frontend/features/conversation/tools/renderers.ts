/**
 * The tool rows' extension points, shared by the Pi transcript and the chat: native React UI for a tool, and its display name.
 *
 * - A tool renderer replaces the default tool row, wraps it (`next()`), or passes per call by returning `next()`
 * - Renderers of one tool chain newest first: `next()` is the one registered before, the last `next()` the default row (`ToolCallRow.tsx`)
 * - `placement: "standalone"` keeps the tool's calls outside the folded "Process details" (todos, plans, questions), the newest registration decides
 * - Registrations are side effects of importing the registering module; a feature registers its own tools, the app shell loads the ones another feature's rows need
 *
 * After pi-web.dev's `resolve(toolName, next)` chain.
 */
import type { LucideIcon } from "lucide-react"
import type { ReactNode } from "react"
import type { ToolCall, ToolRun } from "./types"

export type ToolRenderer = (call: ToolCall, run: ToolRun | undefined, next: () => ReactNode) => ReactNode
export type ToolPlacement = "inline" | "standalone"
/** How a row shows its tool instead of the raw name and category icon. */
export interface ToolDisplay {
  label: string
  icon: LucideIcon
}

interface ToolRegistration {
  render: ToolRenderer
  placement: ToolPlacement
}

const toolRenderers = new Map<string, ToolRegistration[]>()

/** Add `registration` to `name`'s chain; the returned function removes exactly it. */
export function register<T>(registry: Map<string, T[]>, name: string, registration: T): () => void {
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
