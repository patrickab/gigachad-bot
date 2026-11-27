/**
 * The tool rows' extension points, shared by the Pi transcript and the chat: native React UI for a tool, and its display name.
 *
 * - A tool renderer replaces the default tool row, wraps it (`next()`), or passes per call by returning `next()`
 * - Registrations are side effects of importing the registering module; a feature registers its own tools, the app shell loads the ones another feature's rows need
 *
 * After pi-web.dev's `resolve(toolName, next)` chain.
 */
import type { LucideIcon } from "lucide-react"
import type { ReactNode } from "react"
import type { ToolCall, ToolRun } from "./types"

export type ToolRenderer = (call: ToolCall, run: ToolRun | undefined, next: () => ReactNode) => ReactNode
/** How a row shows its tool instead of the raw name and category icon. */
export interface ToolDisplay {
  label: string
  icon: LucideIcon
}

const toolRenderers = new Map<string, ToolRenderer>()

/** Render `toolName`'s calls with `render`, in place of an earlier registration of that name. */
export function registerToolRenderer(toolName: string, render: ToolRenderer): void {
  toolRenderers.set(toolName, render)
}

/** `call` through its tool's renderer, which gets `fallback` (the default row) as `next`; `fallback()` without one. */
export function renderToolCall(call: ToolCall, run: ToolRun | undefined, fallback: () => ReactNode): ReactNode {
  const render = toolRenderers.get(call.name)
  return render ? render(call, run, fallback) : fallback()
}
