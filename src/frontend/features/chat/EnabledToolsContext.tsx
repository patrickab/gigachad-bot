"use client"

import { createContext, useCallback, useEffect, useMemo, useState, useSyncExternalStore, type ReactNode } from "react"
import { useHostAccess } from "@/contexts/SettingsContext"
import { useRequiredContext } from "@/hooks/useRequiredContext"
import { readStorage, writeStorage } from "@/lib/storage"
import { availableTools, TOOLS } from "@/features/chat/tools"
import type { ToolName } from "@/lib/domain/types"

const TOOLS_STORAGE_KEY = "gigachad-enabled-tools"

interface EnabledToolsState {
  /** Tool names the model may call on the next send, among the tools this login may enable. */
  enabledTools: ToolName[]
  toggleTool: (name: ToolName) => void
  /** Deep research gates its own settings in the options menu. */
  researchEnabled: boolean
}

const EnabledToolsContext = createContext<EnabledToolsState | null>(null)

export function useEnabledTools(): EnabledToolsState {
  return useRequiredContext(EnabledToolsContext, "useEnabledTools must be used within EnabledToolsProvider")
}

/** Storage never changes under a mounted provider, so there is nothing to subscribe to. */
function subscribeNever(): () => void {
  return () => {}
}

function readSavedTools(): string | null {
  return readStorage(TOOLS_STORAGE_KEY)
}

function readServerSavedTools(): string | null {
  return null
}

function defaultTools(): ToolName[] {
  return TOOLS.filter((tool) => tool.defaultEnabled).map((tool) => tool.name)
}

/** Tools added after lists were saved as a bare array of names, which recorded no `known`. */
const ADDED_AFTER_BARE_LISTS: readonly ToolName[] = ["start_learn_session"]

/** What storage holds: the enabled tools and every tool the list's owner could choose from when it was saved. */
interface SavedTools {
  enabled: string[]
  known: string[]
}

function readSaved(saved: string): SavedTools {
  const parsed = JSON.parse(saved) as string[] | SavedTools
  if (!Array.isArray(parsed)) return parsed
  return { enabled: parsed, known: TOOLS.map((tool) => tool.name).filter((name) => !ADDED_AFTER_BARE_LISTS.includes(name)) }
}

/**
 * Parses the saved list, keeping defaults when storage is empty or corrupt. A default-on tool the list did not know
 * yet starts enabled; the next save records it as known, so a later opt-out sticks.
 */
function parseSavedTools(saved: string | null): ToolName[] {
  if (!saved) return defaultTools()
  try {
    const { enabled, known } = readSaved(saved)
    return TOOLS
      .filter((tool) => enabled.includes(tool.name) || (tool.defaultEnabled && !known.includes(tool.name)))
      .map((tool) => tool.name)
  } catch {
    return defaultTools()
  }
}

function savedTools(enabled: ToolName[]): string {
  return JSON.stringify({ enabled, known: TOOLS.map((tool) => tool.name) } satisfies SavedTools)
}

export function EnabledToolsProvider({ children }: { children: ReactNode }) {
  // Restored once from storage (a per-browser choice, matching every other composer preference);
  // the server snapshot is empty so the first paint uses the safe defaults.
  const saved = useSyncExternalStore(subscribeNever, readSavedTools, readServerSavedTools)
  const [chosen, setChosen] = useState<ToolName[] | null>(null)
  // The choice is kept as made, a login only ever sends the tools it may be offered.
  const hostAccess = useHostAccess()
  const enabledTools = useMemo(() => {
    const available = new Set(availableTools(hostAccess).map((tool) => tool.name))
    return (chosen ?? parseSavedTools(saved)).filter((name) => available.has(name))
  }, [chosen, saved, hostAccess])

  useEffect(() => {
    if (chosen) writeStorage(TOOLS_STORAGE_KEY, savedTools(chosen))
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
