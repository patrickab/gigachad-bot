"use client"

import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react"
import { Globe, LineChart, Network, NotebookPen, Search, Sigma, Terminal, Workflow, type LucideIcon } from "lucide-react"
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
  { name: "diagram", selectorLabel: "Diagram", shortLabel: "Diagram", cardLabel: "Diagram", icon: Workflow, defaultEnabled: true },
  { name: "mindmap", selectorLabel: "Mind Map", shortLabel: "Mind map", cardLabel: "Mind map", icon: Network, defaultEnabled: false },
  { name: "latex_ocr", selectorLabel: "LaTeX OCR", shortLabel: "LaTeX OCR", cardLabel: "LaTeX OCR", icon: Sigma, defaultEnabled: false },
  { name: "workspace_agent", selectorLabel: "Workspace Agent", shortLabel: "Workspace", cardLabel: "Workspace agent", icon: Terminal, defaultEnabled: false },
]

/** Jupyter mode: server-backed per-chat state, never offered through `tools` — the backend
 *  injects the notebook_edit tool itself whenever a notebook exists. The entry is keyed so
 *  saved notebook_edit calls render like any other tool; the composer menu row lives here
 *  too, next to the labels it shares. */
export const NOTEBOOK_TOOL_META: ToolMeta = {
  name: "notebook_edit",
  selectorLabel: "Jupyter Mode",
  shortLabel: "Notebook",
  cardLabel: "Notebook",
  icon: NotebookPen,
  defaultEnabled: false,
}

/** Keyed for lookup by a saved call's name, which an older chat may no longer match. */
export const TOOL_META: Partial<Record<string, ToolMeta>> = Object.fromEntries(
  [...TOOLS, NOTEBOOK_TOOL_META].map((tool) => [tool.name, tool]),
)

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
  /** Chat whose notebook has been activated; the sidebar reads this to show the notebook
   *  element. Null means no notebook (or activation was reset by a chat switch). */
  notebookChatId: string | null
  setNotebookChatId: (chatId: string | null) => void
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
  // Notebook activation is per chat and server-backed, so it lives in state rather than
  // localStorage; the composer resets it whenever chatId changes.
  const [notebookChatId, setNotebookChatId] = useState<string | null>(null)

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
    notebookChatId,
    setNotebookChatId,
  }


  return (
    <ModeContext.Provider value={value}>
      {children}
    </ModeContext.Provider>
  )
}
