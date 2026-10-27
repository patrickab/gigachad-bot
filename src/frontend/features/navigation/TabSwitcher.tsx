"use client"

import { useCallback, useContext, useEffect, useId, useRef, useState } from "react"
import { ArrowRightLeft, Check, ChevronDown, Plus, X } from "lucide-react"
import { cn } from "@/lib/utils"
import { useTab } from "@/contexts/TabContext"
import { PATH_COLLATOR } from "@/lib/paths"
import { useDisplayName, useProject } from "@/contexts/ProjectContext"
import { useOutsidePress } from "@/hooks/useOutsidePress"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import { TabStripContext, tabLocation } from "./tabs"

const PILL_BUTTON_CLASS = "flex h-8.5 w-8.5 shrink-0 items-center justify-center rounded-full bg-paper text-ink-muted transition-colors hover:bg-surface-elevated hover:text-ink focus:bg-surface-elevated focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-subtle"

function Spinner() {
  return <span className="h-2.5 w-2.5 shrink-0 animate-spin rounded-full border border-ink-faint border-t-ink-muted" />
}

function BreadcrumbLabel({ project, category, file }: { project: string, category: string, file: string }) {
  return <>
    <span className="max-w-24 truncate text-ink-subtle">{project}</span>
    <span aria-hidden="true" className="text-ink-faint">/</span>
    <span className="shrink-0 text-ink-subtle">{category}</span>
    <span aria-hidden="true" className="text-ink-faint">/</span>
    <span className="min-w-0 truncate text-ink">{file}</span>
  </>
}

/**
 * Reveal centered workspace navigation on hover, keyboard focus, or while its menu is open.
 *
 * - Trigger is as wide as the widest open tab label and centered alone, so switching never moves it or its buttons
 */
export function TabSwitcher() {
  const strip = useContext(TabStripContext)
  const { tab, isActive } = useTab()
  const { projects } = useProject()
  const shownName = useDisplayName()
  const activityDescriptionId = useId()
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState("")
  const [showSearch, setShowSearch] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)
  const triggerRef = useRef<HTMLButtonElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  const close = useCallback(() => {
    setOpen(false)
    setQuery("")
  }, [])
  useOutsidePress(rootRef, close, { active: open })
  useEscapeLayer(() => {
    close()
    triggerRef.current?.focus()
  }, open)
  useEffect(() => {
    if (!open || !isActive) return
    const panel = panelRef.current
    const current = panel?.querySelector<HTMLElement>("[aria-current=page]")
    current?.scrollIntoView?.({ block: "nearest" })
    const input = panel?.querySelector<HTMLInputElement>("input")
    const focusTarget = input ?? current
    focusTarget?.focus()
  }, [open, isActive])
  if (!strip) return null

  const projectName = (slug: string | null) => projects.find((project) => project.slug === slug)?.name ?? slug ?? "general"
  const location = tabLocation(tab, shownName)
  const breadcrumb = `${projectName(tab.projectSlug)} / ${location.category} / ${location.file}`
  const search = query.trim().toLowerCase()
  const orderedTabs = strip.tabs.map((item) => ({ item, project: projectName(item.projectSlug), ...tabLocation(item, shownName) }))
    .sort((left, right) => Number(left.project === "general") - Number(right.project === "general")
      || PATH_COLLATOR.compare(left.project, right.project)
      || PATH_COLLATOR.compare(left.category, right.category)
      || PATH_COLLATOR.compare(left.file, right.file)
      || left.item.id.localeCompare(right.item.id))
  const visibleTabs = orderedTabs.filter((entry) => `${entry.project} ${entry.category} ${entry.file}`.toLowerCase().includes(search))
  const following = orderedTabs[(orderedTabs.findIndex(({ item }) => item.id === strip.activeTabId) + 1) % orderedTabs.length].item

  return <div ref={rootRef} data-tab-switcher data-open={open} className="pointer-events-auto flex w-[32rem] max-w-full flex-col items-center opacity-0 transition-opacity duration-160 [&:hover]:opacity-100 focus-within:opacity-100 data-[open=true]:opacity-100 motion-reduce:transition-none"
    onBlur={(event) => { if (!event.currentTarget.contains(event.relatedTarget)) close() }}>
    <div className="relative flex min-w-0 max-w-[calc(100%_-_9.5rem)]">
      <button ref={triggerRef} type="button" aria-label={`Switch tabs: ${breadcrumb}`} aria-haspopup="dialog" aria-expanded={open}
        aria-describedby={activityDescriptionId}
        onClick={() => {
          if (open) close()
          else {
            setShowSearch(strip.tabs.length > 8)
            setOpen(true)
          }
        }}
        className="flex h-8.5 max-w-full min-w-0 items-center gap-1.5 rounded-full bg-paper px-2 text-xs text-ink-muted transition-colors hover:bg-surface-elevated hover:text-ink focus:bg-surface-elevated focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-subtle">
        <span className="grid min-w-0 grid-cols-[minmax(0,max-content)]">
          {orderedTabs.filter(({ item }) => item.id !== tab.id).map(({ item, project, category, file }) =>
            <span key={item.id} aria-hidden="true" className="invisible flex min-w-0 items-center gap-1.5 whitespace-nowrap [grid-area:1/1]">
              <BreadcrumbLabel project={project} category={category} file={file} />
            </span>)}
          <span data-workspace-breadcrumb className="flex min-w-0 items-center gap-1.5 whitespace-nowrap [grid-area:1/1]">
            <BreadcrumbLabel project={projectName(tab.projectSlug)} category={location.category} file={location.file} />
          </span>
        </span>
        <ChevronDown aria-hidden="true" className="h-3 w-3 shrink-0 text-ink-subtle" />
      </button>
      <div className="absolute inset-y-0 left-full ml-1 flex items-center gap-1">
        <button type="button" aria-label="Next tab" disabled={strip.tabs.length < 2}
          onClick={() => {
            close()
            strip.select(following.id)
          }}
          className={`${PILL_BUTTON_CLASS} disabled:cursor-default disabled:text-ink-faint`}>
          <ArrowRightLeft aria-hidden="true" className="h-3 w-3" />
        </button>
        <button type="button" data-new-chat aria-label="New chat" onClick={() => {
          close()
          strip.add()
        }} className={PILL_BUTTON_CLASS}>
          <Plus aria-hidden="true" className="h-3 w-3" />
        </button>
      </div>
    </div>
    <span id={activityDescriptionId} className="sr-only">{strip.labelOf(tab)}</span>
    {open && isActive && <div ref={panelRef} role="dialog" aria-label="Open tabs"
      className="absolute left-1/2 top-full z-50 mt-1 flex w-max max-w-[min(36rem,100%)] -translate-x-1/2 flex-col overflow-hidden rounded-xl border border-divider bg-surface shadow-[var(--shadow-lg)]">
      {showSearch && <div className="border-b border-divider p-2">
        <input aria-label="Search open tabs" placeholder="Find a tab…" value={query} onChange={(event) => setQuery(event.target.value)}
          className="w-full rounded-md bg-paper px-2 py-1.5 text-xs text-ink outline-none focus-visible:ring-1 focus-visible:ring-ink-faint" />
      </div>}
      <div role="group" aria-label="Workspaces" className="max-h-[min(28rem,60vh)] overflow-y-auto p-1.5">
          {visibleTabs.map(({ item, project, category, file }) => {
            const active = item.id === strip.activeTabId
            return <div key={item.id} className={cn("group/tab flex items-center rounded-md transition-colors duration-160 motion-reduce:transition-none", active ? "bg-surface-elevated" : "hover:bg-surface-elevated/60")}>
              <button type="button" aria-label={`Switch to ${project} / ${category} / ${file}`} aria-current={active ? "page" : undefined}
                aria-describedby={`${activityDescriptionId}-${item.id}`}
                onClick={() => {
                  close()
                  strip.select(item.id)
                }}
                className="flex min-w-0 flex-1 items-center gap-2 rounded-md px-2 py-2 text-left text-xs text-ink-muted focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-subtle">
                <span className="min-w-0 flex-1 truncate whitespace-nowrap"><span className="text-ink-subtle">{project} / {category} / </span><span className={active ? "text-ink" : undefined}>{file}</span></span>
                <span id={`${activityDescriptionId}-${item.id}`} className="sr-only">{strip.labelOf(item)}</span>
                {strip.isLoading(item) && <Spinner />}
                {active && <Check aria-hidden="true" className="h-3 w-3 shrink-0 text-ink-subtle" />}
              </button>
              {strip.tabs.length > 1 && <button type="button" aria-label={`Close ${project} / ${category} / ${file}`}
                onClick={() => {
                  if (active) close()
                  else {
                    const target = panelRef.current?.querySelector<HTMLInputElement>("input") ?? triggerRef.current
                    target?.focus()
                  }
                  strip.close(item.id)
                }}
                className="mr-1 rounded p-1 text-ink-faint opacity-0 hover:text-ink group-hover/tab:opacity-100 group-focus-within/tab:opacity-100 focus-visible:opacity-100">
                <X aria-hidden="true" className="h-3 w-3" />
              </button>}
            </div>
          })}
        {visibleTabs.length === 0 && <p className="px-2 py-3 text-xs text-ink-subtle">No matching tabs</p>}
      </div>
    </div>}
  </div>
}
