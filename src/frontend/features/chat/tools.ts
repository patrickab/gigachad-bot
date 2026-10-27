import { Globe, LineChart, Network, Search, Sigma, Terminal, Workflow, Wrench, type LucideIcon } from "lucide-react"
import type { ToolName } from "@/lib/domain/types"

interface ToolMeta {
  name: ToolName
  /** Entry in the composer's tool menu. */
  selectorLabel: string
  /** Heading above the tool call in the transcript. */
  cardLabel: string
  icon: LucideIcon
  defaultEnabled: boolean
}

/** The one place tool labels, icons and defaults live: composer, tool card and mode label all
 *  read this table, in this order. The workspace agent stays opt-in because it executes code
 *  and persists state. */
export const TOOLS: readonly ToolMeta[] = [
  { name: "deep_research", selectorLabel: "Deep Research", cardLabel: "Deep research", icon: Search, defaultEnabled: true },
  { name: "web_search", selectorLabel: "Web Search", cardLabel: "Web search", icon: Globe, defaultEnabled: true },
  { name: "sandbox_plot", selectorLabel: "Interactive Plot", cardLabel: "Interactive plot", icon: LineChart, defaultEnabled: true },
  { name: "diagram", selectorLabel: "Diagram", cardLabel: "Diagram", icon: Workflow, defaultEnabled: true },
  { name: "mindmap", selectorLabel: "Mind Map", cardLabel: "Mind map", icon: Network, defaultEnabled: false },
  { name: "latex_ocr", selectorLabel: "LaTeX OCR", cardLabel: "LaTeX OCR", icon: Sigma, defaultEnabled: false },
  { name: "workspace_agent", selectorLabel: "Workspace Agent", cardLabel: "Workspace agent", icon: Terminal, defaultEnabled: false },
]

/** Keyed for lookup by a saved call's name, which an older chat may no longer match. */
const TOOL_META: Partial<Record<string, ToolMeta>> = Object.fromEntries(TOOLS.map((tool) => [tool.name, tool]))

/** Icon and card label of a saved call's tool. Unknown saved names keep the raw name and a neutral icon. */
export function toolDisplay(name: string): { icon: LucideIcon; label: string } {
  const meta = TOOL_META[name]
  return { icon: meta?.icon ?? Wrench, label: meta?.cardLabel ?? name }
}
