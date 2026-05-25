"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { Plus, X } from "lucide-react"
import { cn } from "@/lib/utils"

export interface Tab {
  id: string
  name: string | null
}

let tabIdCounter = 0

function nextId(): string {
  return `tab-${++tabIdCounter}`
}

interface TabManagerProps {
  renderContent: (tab: Tab, onModeLabel: (label: string) => void) => React.ReactNode
}

const INITIAL_TAB: Tab = { id: nextId(), name: null }

export function TabManager({ renderContent }: TabManagerProps) {
  const [tabs, setTabs] = useState<Tab[]>([INITIAL_TAB])
  const [activeTab, setActiveTab] = useState(INITIAL_TAB.id)
  const [editingTab, setEditingTab] = useState<string | null>(null)
  const [modeLabels, setModeLabels] = useState<Record<string, string>>({})
  const [customNames, setCustomNames] = useState<Record<string, string | null>>({})
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (editingTab) inputRef.current?.focus()
  }, [editingTab])

  const addTab = useCallback(() => {
    const tab: Tab = { id: nextId(), name: null }
    setTabs((prev) => [...prev, tab])
    setActiveTab(tab.id)
  }, [])

  const closeTab = useCallback((id: string) => {
    setTabs((prev) => {
      if (prev.length <= 1) return prev
      const idx = prev.findIndex((t) => t.id === id)
      const next = prev.filter((t) => t.id !== id)
      if (id === activeTab) {
        const targetIdx = Math.max(0, idx - 1)
        setActiveTab(next[targetIdx]?.id ?? "")
      }
      return next
    })
  }, [activeTab])

  const closeActiveTab = useCallback(() => {
    closeTab(activeTab)
  }, [activeTab, closeTab])

  const switchToNextTab = useCallback(() => {
    setActiveTab((current) => {
      const idx = tabs.findIndex((t) => t.id === current)
      const next = (idx + 1) % tabs.length
      return tabs[next].id
    })
  }, [tabs])

  const switchToPrevTab = useCallback(() => {
    setActiveTab((current) => {
      const idx = tabs.findIndex((t) => t.id === current)
      const prev = (idx - 1 + tabs.length) % tabs.length
      return tabs[prev].id
    })
  }, [tabs])

  useEffect(() => {
    function handler(e: KeyboardEvent) {
      if (e.altKey && e.key === "t") {
        e.preventDefault()
        addTab()
      }
      if (e.altKey && e.key === "w") {
        e.preventDefault()
        closeActiveTab()
      }
      if (e.key === "Tab" && !e.altKey && !e.ctrlKey && !e.metaKey) {
        const target = e.target as HTMLElement
        const isEditable =
          target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.isContentEditable
        if (!isEditable) {
          e.preventDefault()
          if (e.shiftKey) {
            switchToPrevTab()
          } else {
            switchToNextTab()
          }
        }
      }
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [addTab, closeActiveTab, switchToNextTab, switchToPrevTab])

  const startRename = useCallback((id: string) => {
    setEditingTab(id)
  }, [])

  const commitRename = useCallback((id: string, value: string) => {
    setEditingTab(null)
    setCustomNames((prev) => {
      const next = { ...prev }
      if (value.trim()) {
        next[id] = value.trim()
      } else {
        next[id] = null
      }
      return next
    })
    setTabs((prev) =>
      prev.map((t) => (t.id === id ? { ...t, name: value.trim() || null } : t))
    )
  }, [])

  const updateModeLabel = useCallback((id: string, label: string) => {
    setModeLabels((prev) => (prev[id] === label ? prev : { ...prev, [id]: label }))
  }, [])

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent, id: string, value: string) => {
      if (e.key === "Enter") commitRename(id, value)
      if (e.key === "Escape") setEditingTab(null)
    },
    [commitRename]
  )

  function getTabDisplayName(tab: Tab): string {
    const custom = customNames[tab.id]
    if (custom !== undefined && custom !== null) return custom
    return modeLabels[tab.id] || "Chat"
  }

  return (
    <div className="flex flex-col h-screen overflow-hidden">
      <div className="h-8 shrink-0 flex items-center border-b border-zinc-800/50 bg-zinc-950 select-none">
        <div className="flex-1 flex items-center h-full min-w-0">
          {tabs.map((tab) => {
            const isActive = tab.id === activeTab
            const isEditing = tab.id === editingTab
            const displayName = getTabDisplayName(tab)

            return (
              <div
                key={tab.id}
                className={cn(
                  "flex-1 min-w-0 h-full flex items-center justify-between px-3 gap-1 border-r border-zinc-800/50 cursor-pointer transition-colors text-xs whitespace-nowrap",
                  isActive
                    ? "bg-zinc-900/50 text-zinc-200"
                    : "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900/30"
                )}
                onClick={() => setActiveTab(tab.id)}
                onContextMenu={(e) => { e.preventDefault(); closeTab(tab.id) }}
                onDoubleClick={() => startRename(tab.id)}
              >
                {isEditing ? (
                  <input
                    ref={inputRef}
                    autoFocus
                    defaultValue={displayName}
                    className="flex-1 bg-transparent outline-none text-zinc-200 text-xs min-w-0"
                    onBlur={(e) => commitRename(tab.id, e.target.value)}
                    onKeyDown={(e) => handleKeyDown(e, tab.id, (e.target as HTMLInputElement).value)}
                    onClick={(e) => e.stopPropagation()}
                  />
                ) : (
                  <span className="truncate">{displayName}</span>
                )}
                {tabs.length > 1 && (
                  <button
                    className="shrink-0 p-0.5 rounded hover:bg-zinc-700/50 text-zinc-600 hover:text-zinc-300"
                    onClick={(e) => {
                      e.stopPropagation()
                      closeTab(tab.id)
                    }}
                  >
                    <X className="h-3 w-3" />
                  </button>
                )}
              </div>
            )
          })}
        </div>
        <button
          className="shrink-0 h-full px-2.5 flex items-center justify-center text-zinc-500 hover:text-zinc-200 hover:bg-zinc-800/50 transition-colors"
          onClick={addTab}
        >
          <Plus className="h-3.5 w-3.5" />
        </button>
      </div>
      <div className="flex-1 min-h-0 overflow-hidden relative">
        {tabs.map((tab) => {
          const hook = (label: string) => updateModeLabel(tab.id, label)
          return (
            <div
              key={tab.id}
              className="h-full"
              style={{ display: tab.id === activeTab ? undefined : "none" }}
            >
              {renderContent(tab, hook)}
            </div>
          )
        })}
      </div>
    </div>
  )
}
