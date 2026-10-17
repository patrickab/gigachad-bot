"use client"

import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react"
import { TOOLS } from "@/features/chat/tools"
import type { ToolName } from "@/lib/domain/types"

const TOOLS_STORAGE_KEY = "gigachad-enabled-tools"

interface EnabledToolsState {
  /** Tool names the model may call on the next send. */
  enabledTools: ToolName[]
  toggleTool: (name: ToolName) => void
  /** Deep research gates its own settings in the options menu. */
  researchEnabled: boolean
}

const EnabledToolsContext = createContext<EnabledToolsState | null>(null)

export function useEnabledTools(): EnabledToolsState {
  const ctx = useContext(EnabledToolsContext)
  if (!ctx) throw new Error("useEnabledTools must be used within EnabledToolsProvider")
  return ctx
}

export function EnabledToolsProvider({ children }: { children: ReactNode }) {
  const [enabledTools, setEnabledTools] = useState<ToolName[]>(() => TOOLS.filter((tool) => tool.defaultEnabled).map((tool) => tool.name))

  // Restored once on mount so the first paint uses the safe defaults above; a per-browser
  // choice, matching every other composer preference (theme).
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

  const value: EnabledToolsState = {
    enabledTools,
    toggleTool,
    researchEnabled: enabledTools.includes("deep_research"),
  }

  return (
    <EnabledToolsContext.Provider value={value}>
      {children}
    </EnabledToolsContext.Provider>
  )
}
