"use client"

import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react"
import { Globe, LineChart, Search, Terminal, type LucideIcon } from "lucide-react"
import type { ToolName } from "@/lib/types"

/** OCR still hijacks the composer; web search, deep research, and sandbox plots are tools the
 *  model calls itself. */
export type AppMode = "chat" | "ocr"

export interface ToolMeta {
  name: ToolName
  /** Entry in the composer's tool menu. */
  selectorLabel: string
  /** Text of the composer pill once the tool is enabled. */
  shortLabel: string
  /** Heading above the tool call in the transcript. */
  cardLabel: string
  icon: LucideIcon
  defaultEnabled: boolean
}

/** The one place tool labels, icons and defaults live: composer, tool card and mode label all
 *  read this table, in this order. The workspace agent stays opt-in because it executes code
 *  and persists state. */
export const TOOLS: readonly ToolMeta[] = [
  { name: "deep_research", selectorLabel: "Deep Research", shortLabel: "Research", cardLabel: "Deep research", icon: Search, defaultEnabled: true },
  { name: "web_search", selectorLabel: "Web Search", shortLabel: "Search", cardLabel: "Web search", icon: Globe, defaultEnabled: true },
  { name: "sandbox_plot", selectorLabel: "Interactive Plot", shortLabel: "Plot", cardLabel: "Interactive plot", icon: LineChart, defaultEnabled: true },
  { name: "workspace_agent", selectorLabel: "Workspace Agent", shortLabel: "Workspace", cardLabel: "Workspace agent", icon: Terminal, defaultEnabled: false },
]

/** Keyed for lookup by a saved call's name, which an older chat may no longer match. */
export const TOOL_META: Partial<Record<string, ToolMeta>> = Object.fromEntries(TOOLS.map((tool) => [tool.name, tool]))

const TOOLS_STORAGE_KEY = "gigachad-enabled-tools"

export interface ModeState {
  mode: AppMode
  /** Tool names the model may call on the next send. */
  enabledTools: ToolName[]
  toggleTool: (name: ToolName) => void
  /** Search and research gate their own settings in the options menu. */
  researchEnabled: boolean
  searchEnabled: boolean
  ocrEnabled: boolean
  toggleOCR: () => void
  setMode: (mode: AppMode) => void
}

const ModeContext = createContext<ModeState | null>(null)

export function useModeState(): ModeState {
  const ctx = useContext(ModeContext)
  if (!ctx) throw new Error("useModeState must be used within ModeProvider")
  return ctx
}

export function ModeProvider({ children }: { children: ReactNode }) {
  const [mode, setMode] = useState<AppMode>("chat")
  const [enabledTools, setEnabledTools] = useState<ToolName[]>(() => TOOLS.filter((tool) => tool.defaultEnabled).map((tool) => tool.name))

  // Restored once on mount so the first paint uses the safe defaults above; a per-browser
  // choice, matching every other composer preference (theme, transparent background).
  useEffect(() => {
    try {
      const saved = window.localStorage.getItem(TOOLS_STORAGE_KEY)
      if (saved) {
        const names = JSON.parse(saved) as string[]
        const valid = new Set(TOOLS.map((tool) => tool.name))
        setEnabledTools(names.filter((name): name is ToolName => valid.has(name as ToolName)))
      }
    } catch { /* corrupt or unavailable storage — keep defaults */ }
  }, [])

  useEffect(() => {
    try { window.localStorage.setItem(TOOLS_STORAGE_KEY, JSON.stringify(enabledTools)) } catch { /* quota */ }
  }, [enabledTools])

  // Tools are independent: enabling one must not disable another, because a chat can call a
  // different tool on every turn.
  const toggleTool = useCallback((name: ToolName) => {
    setEnabledTools((prev) => (prev.includes(name) ? prev.filter((t) => t !== name) : [...prev, name]))
  }, [])

  const toggleOCR = useCallback(() => {
    setMode((prev) => prev === "ocr" ? "chat" : "ocr")
  }, [])

  const value: ModeState = {
    mode,
    enabledTools,
    toggleTool,
    researchEnabled: enabledTools.includes("deep_research"),
    searchEnabled: enabledTools.includes("web_search"),
    ocrEnabled: mode === "ocr",
    toggleOCR,
    setMode,
  }

  return (
    <ModeContext.Provider value={value}>
      {children}
    </ModeContext.Provider>
  )
}
