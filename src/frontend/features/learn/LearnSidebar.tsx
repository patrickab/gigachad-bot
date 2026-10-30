"use client"

import { useEffect, useRef, useState, type KeyboardEvent } from "react"
import { Activity, Dumbbell, Ellipsis, GraduationCap, Network, Pencil, Plus, Search, Settings2 } from "lucide-react"
import { FileTree, type FileTreeItem } from "@/components/ui/file-tree/FileTree"
import { usePathTree } from "@/components/ui/file-tree/useFileTree"
import { ErrorText } from "@/components/ui/ErrorText"
import { PillTabs } from "@/components/ui/PillTabs"
import { SaveAsModal } from "@/components/ui/SaveAsModal"
import { useTab } from "@/contexts/TabContext"
import { useUndoDelete } from "@/contexts/UndoDeleteContext"
import { useAsyncAction } from "@/hooks/useAsyncAction"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import { useEventListener } from "@/hooks/useEventListener"
import { useStoredString } from "@/hooks/useStoredString"
import { exportLearnSession } from "@/lib/backend/api"
import type { LearnSession, LearnView } from "@/lib/domain/types"
import { errorMessage } from "@/lib/utils"
import type { LearnChromeKey, LearnFrameMessageOf, LearnPanel, LearnSessionStatus } from "@gigachad/chat-ui/learn-protocol"
import { learnItem, type OpenItem } from "./items"
import { useLearn, useLearnDrive, useLearnFrameEvent, useLearnFrameState } from "./LearnProvider"
import { useLearnPanels } from "./useLearnPanels"

// The FileTree's own hover-revealed row actions, for the section's extra header actions.
const HEADER_ACTIONS_CLASS = "flex shrink-0 items-center gap-0.5 max-w-0 overflow-hidden opacity-0 transition-opacity group-hover/row:max-w-40 group-hover/row:opacity-100 group-focus-within/row:max-w-40 group-focus-within/row:opacity-100"
const HEADER_ACTION_CLASS = "shrink-0 rounded p-1 text-ink-faint hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"
// The session menu's entries, InlineConfirm's answer buttons.
const MENU_ACTION_CLASS = "rounded px-1 text-ink-muted hover:bg-hover hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"

type SessionMeta = LearnFrameMessageOf<"session-meta">

/** LearnHarness's sidebar kind filter, kept across reloads like upstream's (`lh:session-kind-filter`). */
export type LearnKindFilter = "teach" | "practice" | "all"
const KIND_FILTER_KEY = "learn_session_kind_filter"
const KIND_FILTERS: readonly { value: LearnKindFilter; label: string }[] = [
  { value: "teach", label: "Teaching" },
  { value: "practice", label: "Practice" },
  { value: "all", label: "All" },
]

/** The row chip of each state LearnHarness's sidebar row showed, exhaustive over the protocol's states. */
const STATUS_CHIP = {
  compacting: "compacting",
  waiting: "waiting",
  problem: "problem waiting",
  working: "working",
} as const satisfies Record<LearnSessionStatus, string>

/** States that wait on the learner: their chip says for how long, like upstream's "Waiting for input · 5m ago". */
const SHOWS_AGE: Readonly<Record<LearnSessionStatus, boolean>> = { compacting: true, waiting: true, problem: true, working: false }

/** A session's state from the list, the frame's `session-meta` derivation (upstream's `sessionMeta`) for when it sent none. */
function listStatus(session: LearnSession): LearnSessionStatus | null {
  if (session.compacting) return "compacting"
  if (session.pending_question) return "waiting"
  if (session.kind === "practice" && session.pending_problem) return "problem"
  return session.busy ? "working" : null
}

/** How long ago `iso` was, compact: "now", "5m", "3h", "2d", then the date. */
export function sinceLabel(iso: string, now: number = Date.now()): string {
  const minutes = Math.floor((now - Date.parse(iso)) / 60_000)
  if (minutes < 1) return "now"
  if (minutes < 60) return `${minutes}m`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h`
  const days = Math.floor(hours / 24)
  return days < 7 ? `${days}d` : iso.slice(0, 10)
}

/**
 * The chip LearnHarness's hidden chrome showed for a session: its sidebar row's state (with how long it waits), else its
 * title bar's "practice" tag or phase. The frame's `meta` is fresher than the list.
 */
export function sessionChip(session: LearnSession, meta: SessionMeta | undefined, now: number = Date.now()): string | null {
  const status = meta ? meta.status : listStatus(session)
  if (status) {
    const updated = meta?.updatedAt ?? session.updated_at
    return SHOWS_AGE[status] && updated ? `${STATUS_CHIP[status]} ${sinceLabel(updated, now)}` : STATUS_CHIP[status]
  }
  const kind = meta?.kind ?? session.kind
  return kind === "practice" ? "practice" : meta?.phase ?? null
}

/**
 * LearnHarness's session search ranking (`fuzzyScore`): every space-separated token must occur in the title, an exact
 * title beats a prefix beats an early substring, the whole query verbatim adds a bonus. Null: no match.
 */
export function titleScore(query: string, title: string): number | null {
  const tokens = query.trim().toLowerCase().split(/\s+/).filter(Boolean)
  const field = title.toLowerCase()
  let total = 0
  for (const token of tokens) {
    const at = field.indexOf(token)
    if (at === -1) return null
    total += field === token ? 100 : at === 0 ? 60 : 30 - Math.min(at, 27)
  }
  return field.includes(tokens.join(" ")) ? total + 50 : total
}

/** `sessions` matching `query` and `kind`, best match first (ties keep the list's order, most recent first). */
export function filterSessions(sessions: readonly LearnSession[], query: string, kind: LearnKindFilter): LearnSession[] {
  return sessions
    .filter((session) => kind === "all" || (session.kind ?? "teach") === kind)
    .map((session) => ({ session, score: titleScore(query, session.title) }))
    .filter((entry): entry is { session: LearnSession; score: number } => entry.score !== null)
    .sort((a, b) => b.score - a.score)
    .map((entry) => entry.session)
}

/** Save `file` under `filename`, like LearnHarness's own "Export as Markdown" anchor. */
function download(file: Blob, filename: string) {
  const url = URL.createObjectURL(file)
  const anchor = Object.assign(document.createElement("a"), { href: url, download: filename })
  document.body.append(anchor)
  anchor.click()
  anchor.remove()
  URL.revokeObjectURL(url)
}

/** The title filter: its query, kind and the walked match Enter opens. */
export interface LearnFilterState {
  query: string
  setQuery: (query: string | null) => void
  kind: LearnKindFilter
  setKind: (kind: LearnKindFilter) => void
  /** ↑/↓ walk the matches, Enter opens the walked one (the first without a walk). */
  onKeyDown: (event: KeyboardEvent<HTMLInputElement>) => void
  /** Changes whenever the filter is asked for again, the open input takes the focus then. */
  focusRequest: number
}

/** What the Learn section shows and does. */
export interface LearnSectionState {
  /** Session rows (every match while filtering, else those outside projects) and the background agents folder. */
  items: FileTreeItem<string>[]
  /** The title filter, null while it is closed. */
  filter: LearnFilterState | null
  toggleFilter: () => void
  /** A new teaching session (`/chat/new`) in the current tab. */
  openLearn: () => void
  newTeaching: () => void
  newPractice: () => void
  /** LearnHarness's home: its knowledge graph. */
  openGraph: () => void
  /** Opens LearnHarness's settings dialog, absent when the frame runs another protocol version (its own settings stay visible). */
  openSettings?: () => void
  error: string | null
}

/** Props of the learn session rename `SaveAsModal`. */
export interface LearnRenameDialog {
  open: boolean
  onClose: () => void
  initialName?: string
  pending: boolean
  error: string
  onSave: (title: string) => void
}

export interface LearnSidebar {
  /** Project root `root` with an action starting a session that joins the project, and a "learn" folder of its sessions when it has any. */
  withProjectLearn: (root: FileTreeItem<string>, project: { slug: string; name: string }) => FileTreeItem<string>
  section: LearnSectionState
  renameDialog: LearnRenameDialog
}

/** The shell sidebar the learn section sits in: LearnHarness's ⌘B toggles it, ⌘⇧F opens it on the filter. */
export interface LearnSidebarFrame {
  collapsed: boolean
  toggle: () => void
}

/** A session's menu under its row: upstream's hidden session menu and title bar entries. */
function SessionMenu({ entries, onClose }: { entries: readonly { label: string; run: () => void }[]; onClose: () => void }) {
  const first = useRef<HTMLButtonElement>(null)
  useEffect(() => { first.current?.focus() }, [])
  useEscapeLayer(onClose)
  return <div role="menu" aria-label="Session actions" className="flex flex-wrap items-center gap-1 py-0.5 text-[11px]">
    {entries.map(({ label, run }, index) => (
      <button key={label} ref={index === 0 ? first : undefined} role="menuitem" className={MENU_ACTION_CLASS}
        onClick={() => { onClose(); run() }}>
        {label}
      </button>
    ))}
  </div>
}

/**
 * The learn parts of the sidebar, LearnHarness's own sidebar and title bar being hidden in the frame.
 *
 * - Session rows open in the current tab and show the hidden chrome's chip: the row's state ("waiting 5m"), else the
 *   title bar's practice tag or phase. They rename on double-click, their pencil or F2 (also pressed in the frame), delete
 *   with undo (Knowledge stays), and their menu holds the rest: upstream's Plan popup, coverage with "Extract now",
 *   Markdown export, delete with the graph traces
 * - The Learn section starts teaching and practice sessions, opens the knowledge graph and LearnHarness's settings,
 *   filters by title (LearnHarness's ranking, its Teaching/Practice/All filter, ⌘⇧F in the frame) over every session,
 *   and lists the frame's background agents, whose details, costs and stop buttons are upstream's own section
 * - A tab showing a session deleted here goes to LearnHarness's home, like upstream's sidebar delete
 */
export function useLearnSidebar({ open, projects, sidebar }: {
  open: OpenItem
  projects: readonly { slug: string; name: string }[]
  sidebar: LearnSidebarFrame
}): LearnSidebar {
  const { tab, isActive, setWorkspaceItem } = useTab()
  const { sessions, showInFrame } = useLearn()
  const metas = useLearnFrameState("session-meta")
  const { hasPlan, openPanel } = useLearnPanels()
  const activity = useLearnFrameState("activity")
  const drive = useLearnDrive()
  const { schedule, pending } = useUndoDelete()
  const [query, setQuery] = useState<string | null>(null)
  const [storedKind, storeKind] = useStoredString(KIND_FILTER_KEY)
  const kind: LearnKindFilter = KIND_FILTERS.some((filter) => filter.value === storedKind) ? storedKind as LearnKindFilter : "all"
  const [walk, setWalk] = useState<number | null>(null)
  const [focusRequest, setFocusRequest] = useState(0)
  const [menuFor, setMenuFor] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [renameTarget, setRenameTarget] = useState<LearnSession | null>(null)
  const { run: rename, pending: renamePending, error: renameError, setError: setRenameError } = useAsyncAction(async (target: LearnSession, title: string) => {
    await sessions.rename(target.id, title)
    setRenameTarget(null)
  }, "Could not rename this session")

  const shown = tab.workspaceItem?.kind === "learn" ? tab.workspaceItem.learn : undefined
  const activeSessionId = shown?.kind === "session" ? shown.sessionId : null
  const listed = sessions.sessions.filter((session) => !pending(`learn:${session.id}`))
  const activeSession = sessions.sessions.find((session) => session.id === activeSessionId) ?? null

  // This tab's session was deleted: go home like upstream's sidebar does.
  const deleted = activeSessionId !== null && sessions.missing.has(activeSessionId)
  useEffect(() => {
    if (deleted) setWorkspaceItem(learnItem({ kind: "home" }, tab.projectSlug))
  }, [deleted, setWorkspaceItem, tab.projectSlug])

  const startRename = (session: LearnSession) => {
    setRenameError("")
    setRenameTarget(session)
  }

  const openFilter = () => {
    if (sidebar.collapsed) sidebar.toggle()
    setQuery((current) => current ?? "")
    setFocusRequest((count) => count + 1)
  }

  // The hidden chrome's shortcuts pressed in the frame, handled by the tab in front.
  const chromeKeys: Record<LearnChromeKey, () => void> = {
    "toggle-sidebar": sidebar.toggle,
    "sessions-search": openFilter,
    "rename-session": () => { if (activeSession) startRename(activeSession) },
  }
  useLearnFrameEvent("chrome-key", ({ action }) => {
    if (isActive) chromeKeys[action]()
  })
  // F2 with the shell focused renames the session the tab shows, like upstream's title bar.
  useEventListener("window", "keydown", (event) => {
    if (event.key !== "F2" || event.metaKey || event.ctrlKey || event.altKey || event.shiftKey || !activeSession) return
    event.preventDefault()
    startRename(activeSession)
  }, { active: isActive && activeSession !== null })

  const panel = (session: LearnSession, name: LearnPanel) => () => openPanel(session, name)

  const exportMarkdown = async (session: LearnSession) => {
    setActionError(null)
    try {
      const { filename, markdown } = await exportLearnSession(session.id)
      download(new Blob([markdown], { type: "text/markdown" }), filename)
    } catch (cause) {
      setActionError(errorMessage(cause, "Could not export this session"))
    }
  }

  const menuEntries = (session: LearnSession) => [
    ...hasPlan(session.id) ? [{ label: "Plan", run: panel(session, "plan") }] : [],
    { label: "Coverage", run: panel(session, "coverage") },
    { label: "Export as Markdown", run: () => void exportMarkdown(session) },
    { label: "Delete with graph traces", run: () => schedule(`learn:${session.id}`, `${session.title} and its graph traces`, () => sessions.remove(session.id, "delete-nodes")) },
  ]

  const sessionRow = (session: LearnSession, options: { prefix?: string; walked?: boolean } = {}): FileTreeItem<string> => {
    const chip = sessionChip(session, metas[session.id])
    return {
      id: `learn:${session.id}`, label: session.title, prefix: options.prefix, type: "element",
      isActive: options.walked ?? session.id === activeSessionId,
      meta: chip ? <span className="text-ink-faint">{chip}</span> : undefined,
      onSelect: () => open(learnItem({ kind: "session", sessionId: session.id }, session.project_slug)),
      onDoubleClick: () => startRename(session),
      actions: [
        { label: `Rename ${session.title}`, icon: Pencil, onClick: () => startRename(session) },
        { label: `More actions for ${session.title}`, icon: Ellipsis, onClick: () => setMenuFor((current) => current === session.id ? null : session.id) },
      ],
      onDelete: () => schedule(`learn:${session.id}`, session.title, () => sessions.remove(session.id)),
      detail: menuFor === session.id ? <SessionMenu entries={menuEntries(session)} onClose={() => setMenuFor(null)} /> : undefined,
    }
  }

  const withProjectLearn = (root: FileTreeItem<string>, project: { slug: string; name: string }): FileTreeItem<string> => {
    const own = listed.filter((session) => session.project_slug === project.slug)
    const action = { label: `New learn session in ${project.name}`, icon: GraduationCap, onClick: () => open(learnItem({ kind: "new" }, project.slug), true) }
    const folder: FileTreeItem<string> = {
      id: `project:${project.slug}:learn`, label: "learn", type: "folder", children: own.map((session) => sessionRow(session)),
    }
    return { ...root, actions: [...root.actions ?? [], action], children: own.length > 0 ? [...root.children ?? [], folder] : root.children }
  }

  const projectName = (slug: string | null) => projects.find((project) => project.slug === slug)?.name
  // Filtering searches every session, a project's sessions name their project. The walked match (else the first) is highlighted.
  const matches = query === null ? [] : filterSessions(listed, query, kind)
  const walked = matches.length === 0 ? null : (walk ?? 0) % matches.length
  const rows = query !== null
    ? matches.map((session, index) => sessionRow(session, { prefix: projectName(session.project_slug), walked: index === walked }))
    : listed.filter((session) => session.project_slug === null).map((session) => sessionRow(session))
  const agents = activity?.items ?? []
  const agentsFolder: FileTreeItem<string>[] = agents.length === 0 ? [] : [{
    id: "learn-activity", label: "background agents", type: "folder",
    actions: [{ label: "Show background agents", icon: Activity, onClick: () => showInFrame({ type: "open-panel", panel: "activity", sessionId: null }) }],
    children: agents.map((agent) => {
      const sessionId = agent.sessionId
      return {
        id: `learn-activity:${agent.id}`, label: agent.label, type: "element", badge: agent.status,
        onSelect: sessionId ? () => open(learnItem({ kind: "session", sessionId }, sessions.sessions.find((session) => session.id === sessionId)?.project_slug ?? null)) : undefined,
      }
    }),
  }]

  const changeQuery = (next: string | null) => {
    setQuery(next)
    setWalk(null)
  }
  const onFilterKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.metaKey || event.ctrlKey || event.altKey || event.shiftKey || matches.length === 0) return
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault()
      const step = event.key === "ArrowDown" ? 1 : matches.length - 1
      setWalk((walked ?? 0) + step)
      return
    }
    if (event.key === "Enter" && walked !== null) {
      event.preventDefault()
      const target = matches[walked]
      open(learnItem({ kind: "session", sessionId: target.id }, target.project_slug))
      setWalk(null)
      event.currentTarget.blur()
    }
  }

  const newSession = (view: LearnView) => open(learnItem(view), true)

  return {
    withProjectLearn,
    section: {
      items: [...rows, ...agentsFolder],
      filter: query === null ? null : {
        query, setQuery: changeQuery, kind, setKind: (next) => { storeKind(next); setWalk(null) }, onKeyDown: onFilterKeyDown, focusRequest,
      },
      toggleFilter: () => (query === null ? openFilter() : changeQuery(null)),
      openLearn: () => open(learnItem({ kind: "new" })),
      newPractice: () => newSession({ kind: "new", sessionKind: "practice" }),
      newTeaching: () => newSession({ kind: "new" }),
      openGraph: () => open(learnItem({ kind: "home" })),
      openSettings: drive === "mismatch" ? undefined : () => showInFrame({ type: "open-settings", section: null }),
      error: actionError ?? sessions.error,
    },
    renameDialog: {
      open: renameTarget != null,
      onClose: () => { if (!renamePending) setRenameTarget(null) },
      initialName: renameTarget?.title,
      pending: renamePending,
      error: renameError,
      onSave: (title: string) => { if (renameTarget && !renamePending) void rename(renameTarget, title) },
    },
  }
}

/** Title filter of the Learn section with LearnHarness's kind filter: Escape clears and closes it. */
function LearnFilter({ filter }: { filter: LearnFilterState }) {
  const { isActive } = useTab()
  useEscapeLayer(() => filter.setQuery(null), isActive)
  const input = useRef<HTMLInputElement>(null)
  useEffect(() => { input.current?.focus() }, [filter.focusRequest])
  return <div className="mb-1 flex flex-col gap-1 px-2">
    <input ref={input} type="search" aria-label="Filter learn sessions" placeholder="Filter sessions" value={filter.query}
      onChange={(event) => filter.setQuery(event.target.value)} onKeyDown={filter.onKeyDown}
      className="h-6 w-full bg-transparent text-xs text-ink placeholder:text-ink-faint outline-none" />
    <PillTabs options={KIND_FILTERS} value={filter.kind} onChange={filter.setKind} />
  </div>
}

/** The sidebar's Learn section and the learn session rename dialog. */
export function LearnSection({ section, renameDialog }: { section: LearnSectionState; renameDialog: LearnRenameDialog }) {
  const controller = usePathTree()
  const filtering = section.filter !== null
  return <>
    <FileTree label="Learn" items={section.items} controller={controller}
      labelAction={{ label: "Open learn", onClick: section.openLearn }}
      headerActions={<div className={HEADER_ACTIONS_CLASS}>
        <button aria-label="Filter learn sessions" aria-pressed={filtering} onClick={section.toggleFilter} className={HEADER_ACTION_CLASS}>
          <Search className="h-3 w-3" />
        </button>
        <button aria-label="Open knowledge graph" onClick={section.openGraph} className={HEADER_ACTION_CLASS}>
          <Network className="h-3 w-3" />
        </button>
        {section.openSettings && <button aria-label="Open learn settings" onClick={section.openSettings} className={HEADER_ACTION_CLASS}>
          <Settings2 className="h-3 w-3" />
        </button>}
        <button aria-label="New practice session" onClick={section.newPractice} className={HEADER_ACTION_CLASS}>
          <Dumbbell className="h-3 w-3" />
        </button>
        <button aria-label="New teaching session" onClick={section.newTeaching} className={HEADER_ACTION_CLASS}>
          <Plus className="h-3 w-3" />
        </button>
      </div>}
    />
    {section.filter && <LearnFilter filter={section.filter} />}
    <ErrorText className="px-2 py-1">{section.error}</ErrorText>
    <SaveAsModal {...renameDialog} title="Rename learn session" icon={Pencil} submitLabel="Rename" />
  </>
}
