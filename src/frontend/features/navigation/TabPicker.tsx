"use client"

import { useContext, useLayoutEffect, useRef, useState, type KeyboardEvent as ReactKeyboardEvent } from "react"
import { Plus, X } from "lucide-react"
import { FloatingPicker, isRemoveRowKey, neighbourKey } from "@/components/ui/CommandPicker"
import { useTab, type Tab } from "@/contexts/TabContext"
import { useDisplayName, useProject } from "@/contexts/ProjectContext"
import { useEventListener } from "@/hooks/useEventListener"
import { useResetOnChange } from "@/hooks/useResetOnChange"
import { PATH_COLLATOR } from "@/lib/paths"
import { isAltOnly, isEditableTarget } from "@/lib/utils"
import { TabStripContext, tabLocation, type TabStripValue } from "./tabs"

const GENERAL = "general"

/** Controls a pressed Space acts on itself: buttons, links, checkboxes and the like. */
const SPACE_CONTROLS = "button, a[href], summary, select, [role='button'], [role='checkbox'], [role='switch'], [role='option'], [role='menuitem'], [role='tab'], [data-terminal]"

interface TabEntry { tab: Tab; project: string; category: string; file: string }

function Spinner() {
  return <span className="h-2.5 w-2.5 shrink-0 animate-spin rounded-full border border-ink-faint border-t-ink-muted" />
}

/** A bare Space pressed outside text fields and controls that take it, nothing else took it. */
function isPickerSpace(event: KeyboardEvent): boolean {
  const plain = !event.ctrlKey && !event.metaKey && !event.altKey && !event.shiftKey && !event.isComposing && !event.defaultPrevented
  const target = event.target instanceof Element ? event.target : null
  return plain && event.code === "Space" && !isEditableTarget(target, { includeSelect: true }) && !target?.closest(SPACE_CONTROLS)
}

/** Delete at the search's end, or Alt+W: close the highlighted tab and keep picking. */
function isCloseTabKey(event: ReactKeyboardEvent<HTMLInputElement>): boolean {
  return isRemoveRowKey(event) || (event.key === "w" && isAltOnly(event))
}

/**
 * Open state of the tab picker: a tapped Space opens it.
 *
 * - Opens on release, a press held through a pointer press is the canvas's pan, not a tap
 * - Listens only while its tab is frontmost, and closes when the tab goes to the background
 */
function useSpaceToOpen(isActive: boolean) {
  const [open, setOpen] = useState(false)
  const armed = useRef(false)
  useEventListener("window", "keydown", (event) => {
    if (event.repeat || !isPickerSpace(event)) return
    // No page scroll under the opening picker
    event.preventDefault()
    armed.current = true
  }, { active: isActive && !open })
  useEventListener("window", "pointerdown", () => { armed.current = false }, { active: isActive && !open })
  useEventListener("window", "keyup", (event) => {
    if (event.code !== "Space" || !armed.current) return
    armed.current = false
    setOpen(true)
  }, { active: isActive && !open })
  useResetOnChange(isActive, (active) => { if (!active) setOpen(false) })
  return { open, close: () => setOpen(false) }
}

/** One headed group per project, general last, each sorted by category and natural file name. */
function groupsOf(entries: TabEntry[]) {
  const sorted = [...entries].sort((left, right) => Number(left.project === GENERAL) - Number(right.project === GENERAL)
    || PATH_COLLATOR.compare(left.project, right.project)
    || PATH_COLLATOR.compare(left.category, right.category)
    || PATH_COLLATOR.compare(left.file, right.file)
    || left.tab.id.localeCompare(right.tab.id))
  const groups = new Map<string, { key: string; label: string; items: TabEntry[] }>()
  for (const entry of sorted) {
    const group = groups.get(entry.project) ?? { key: entry.project, label: entry.project, items: [] }
    group.items.push(entry)
    groups.set(entry.project, group)
  }
  return [...groups.values()]
}

/** Fills the preview window, the tab `tabId` is scaled into it exactly as it looks frontmost (`TabManager`). */
function TabPeekSlot({ tabId, peek }: { tabId: string; peek: TabStripValue["peek"] }) {
  const slotRef = useRef<HTMLDivElement>(null)
  useLayoutEffect(() => {
    if (!slotRef.current) return
    peek({ id: tabId, slot: slotRef.current })
    return () => peek(null)
  }, [tabId, peek])
  return <div ref={slotRef} className="h-full w-full" />
}

/** Picker over every open tab, opened by Space: grouped by project, the highlighted background tab shown live in the preview window. */
export function TabPicker() {
  const strip = useContext(TabStripContext)
  const { isActive } = useTab()
  const { projects } = useProject()
  const shownName = useDisplayName()
  const { open, close } = useSpaceToOpen(isActive && strip !== null)
  if (!strip || !open) return null

  const projectName = (slug: string | null) => projects.find((project) => project.slug === slug)?.name ?? slug ?? GENERAL
  const entries = strip.tabs.map((tab) => ({ tab, project: projectName(tab.projectSlug), ...tabLocation(tab, shownName) }))
  const closable = strip.tabs.length > 1

  return <FloatingPicker<TabEntry>
    label="Tabs"
    groups={groupsOf(entries)}
    itemKey={(entry) => entry.tab.id}
    searchPlaceholder="Search tabs…"
    searchText={(entry) => `${entry.project} ${entry.category} ${entry.file} ${strip.labelOf(entry.tab)}`}
    searchAction={(closePicker) => <button type="button" aria-label="New tab" onClick={() => { strip.add(); closePicker() }}
      className="rounded p-1 text-ink-faint hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">
      <Plus aria-hidden className="h-3.5 w-3.5" />
    </button>}
    renderItem={(entry) => <>
      <span className="flex min-w-0 flex-1 items-baseline gap-2">
        <span className="truncate leading-5">{entry.file}</span>
        <span className="shrink-0 text-[10px] text-ink-faint">{entry.category}{entry.tab.id === strip.activeTabId && " · current"}</span>
      </span>
      {strip.isLoading(entry.tab) && <Spinner />}
      {closable && <button type="button" aria-label={`Close ${entry.project} / ${entry.category} / ${entry.file}`}
        onClick={(event) => { event.stopPropagation(); strip.close(entry.tab.id) }}
        className="rounded p-0.5 text-ink-faint opacity-0 hover:text-ink focus-visible:opacity-100 in-data-highlighted:opacity-100">
        <X aria-hidden className="h-3 w-3" />
      </button>}
    </>}
    renderPreview={(entry) => entry.tab.id === strip.activeTabId ? null : <TabPeekSlot tabId={entry.tab.id} peek={strip.peek} />}
    initialKey={strip.activeTabId}
    onKeyDown={(event, entry, list) => {
      const opensTab = event.key === "t" && isAltOnly(event)
      if (!opensTab && !isCloseTabKey(event)) return false
      // Handled here, so `TabManager`'s shortcut never switches away from the tab holding this picker
      event.preventDefault()
      event.stopPropagation()
      if (opensTab) {
        list.highlight(strip.add({ activate: false }))
        return true
      }
      // The current tab holds this picker, it stays
      if (!entry || !closable || entry.tab.id === strip.activeTabId) return true
      const neighbour = neighbourKey(list, entry.tab.id)
      if (neighbour) list.highlight(neighbour)
      strip.close(entry.tab.id)
      return true
    }}
    onPick={(entry) => { strip.select(entry.tab.id); return true }}
    onClose={close}
    panelClassName="w-64"
  />
}
