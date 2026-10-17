"use client"

import { createContext, useCallback, useContext, useEffect, useMemo, useState, useSyncExternalStore, type ReactNode } from "react"
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

/** Storage never changes under a mounted provider, so there is nothing to subscribe to. */
function subscribeNever(): () => void {
  return () => {}
}

function readSavedTools(): string | null {
  try { return window.localStorage.getItem(TOOLS_STORAGE_KEY) } catch { return null }
}

function writeSavedTools(tools: ToolName[]): void {
  try { window.localStorage.setItem(TOOLS_STORAGE_KEY, JSON.stringify(tools)) } catch { /* quota */ }
}

function readServerSavedTools(): string | null {
  return null
}

function defaultTools(): ToolName[] {
  return TOOLS.filter((tool) => tool.defaultEnabled).map((tool) => tool.name)
}

/** Parses the saved list, keeping defaults when storage is empty or corrupt. */
function parseSavedTools(saved: string | null): ToolName[] {
  if (!saved) return defaultTools()
  try {
    const names = JSON.parse(saved) as string[]
    const valid = new Set(TOOLS.map((tool) => tool.name))
    return names.filter((name): name is ToolName => valid.has(name as ToolName))
  } catch {
    return defaultTools()
  }
}

export function EnabledToolsProvider({ children }: { children: ReactNode }) {
  // Restored once from storage (a per-browser choice, matching every other composer preference);
  // the server snapshot is empty so the first paint uses the safe defaults.
  const saved = useSyncExternalStore(subscribeNever, readSavedTools, readServerSavedTools)
  const [chosen, setChosen] = useState<ToolName[] | null>(null)
  const enabledTools = useMemo(() => chosen ?? parseSavedTools(saved), [chosen, saved])

  useEffect(() => {
    if (chosen) writeSavedTools(chosen)
  }, [chosen])

  // Tools are independent: enabling one must not disable another, because a chat can call a
  // different tool on every turn.
  const toggleTool = useCallback((name: ToolName) => {
    setChosen((prev) => {
      const current = prev ?? parseSavedTools(saved)
      return current.includes(name) ? current.filter((t) => t !== name) : [...current, name]
    })
  }, [saved])

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
