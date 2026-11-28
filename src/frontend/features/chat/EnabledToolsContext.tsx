"use client"

import { createContext, useEffect, useMemo, useState, useSyncExternalStore, type ReactNode } from "react"
import { useHostAccess } from "@/contexts/SettingsContext"
import { useTab } from "@/contexts/TabContext"
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

/** The chat a tab shows: a tab's choice lasts while it shows that chat. */
interface TabChat { id: string; chatId: string }
type Choices = ReadonlyMap<string, TabChat & { tools: ToolName[] }>

/** The tools of every tab's chat: its own choice, else the saved one. */
interface EnabledToolsStore {
  enabledFor: (tab: TabChat) => ToolName[]
  toggle: (tab: TabChat, name: ToolName) => void
}

const EnabledToolsContext = createContext<EnabledToolsStore | null>(null)

/** The enabled tools of the enclosing tab's chat, in the app's chrome of the tab in front's: a toggle stays in its chat. */
export function useEnabledTools(): EnabledToolsState {
  const { enabledFor, toggle } = useRequiredContext(EnabledToolsContext, "useEnabledTools must be used within EnabledToolsProvider")
  const { tab: { id, chatId } } = useTab()
  return useMemo<EnabledToolsState>(() => {
    const enabledTools = enabledFor({ id, chatId })
    return { enabledTools, toggleTool: (name) => toggle({ id, chatId }, name), researchEnabled: enabledTools.includes("deep_research") }
  }, [enabledFor, toggle, id, chatId])
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
const ADDED_AFTER_BARE_LISTS: readonly ToolName[] = ["start_learn_session", "memory"]

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
  // Restored from storage (a per-browser choice, matching every other composer preference) whenever the provider renders,
  // so a chat that chose nothing follows the last choice saved; the server snapshot is empty so the first paint uses the safe defaults.
  const saved = useSyncExternalStore(subscribeNever, readSavedTools, readServerSavedTools)
  const [chosen, setChosen] = useState<{ byTab: Choices; last: ToolName[] | null }>({ byTab: new Map(), last: null })
  // The choice is kept as made, a login only ever sends the tools it may be offered.
  const hostAccess = useHostAccess()

  useEffect(() => {
    if (chosen.last) writeStorage(TOOLS_STORAGE_KEY, savedTools(chosen.last))
  }, [chosen.last])

  const value = useMemo<EnabledToolsStore>(() => {
    const available = new Set(availableTools(hostAccess).map((tool) => tool.name))
    const choiceOf = (byTab: Choices, tab: TabChat) => {
      const choice = byTab.get(tab.id)
      return choice?.chatId === tab.chatId ? choice.tools : parseSavedTools(saved)
    }
    return {
      enabledFor: (tab) => choiceOf(chosen.byTab, tab).filter((name) => available.has(name)),
      // Tools are independent: enabling one must not disable another, because a chat can call a different tool on every turn.
      toggle: (tab, name) => setChosen(({ byTab }) => {
        const current = choiceOf(byTab, tab)
        const tools = current.includes(name) ? current.filter((t) => t !== name) : [...current, name]
        return { byTab: new Map(byTab).set(tab.id, { ...tab, tools }), last: tools }
      }),
    }
  }, [chosen.byTab, saved, hostAccess])

  return (
    <EnabledToolsContext.Provider value={value}>
      {children}
    </EnabledToolsContext.Provider>
  )
}
