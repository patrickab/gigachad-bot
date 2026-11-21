"use client"

import { useCallback, useRef, useState } from "react"
import { createPortal } from "react-dom"
import { Check, ChevronDown, ChevronRight } from "lucide-react"
import { CommandPicker, type PickerGroup } from "@/components/ui/CommandPicker"
import { usePillPopover } from "@/components/ui/PillPopover"
import { useHostAccess } from "@/contexts/SettingsContext"
import { useLoadWhile } from "@/hooks/useLoadWhile"
import { fileRootTree, type PiRepository, type PiSessionInfo } from "@/lib/backend/api"
import { mountOf } from "@/lib/domain/files"
import { fuzzyPathScore, rankByScore } from "@/lib/fuzzy"
import { PATH_COLLATOR, baseName, isInside, trimSlash } from "@/lib/paths"
import { compactElapsed } from "@/lib/utils"
import { usePiRepositories, usePiSessions } from "./sessionStore"
import { piSessionRepository, piSessionsInside, piSessionTitle } from "./usePiSidebar"

/** A directory a Pi session can be asked to start in, with the newest coding session that ran in it or below it. */
export interface PiStartDirectory {
  path: string
  label: string
  latest: PiSessionInfo | null
}

type Entry = { kind: "none" } | { kind: "unused"; count: number } | { kind: "folder"; directory: PiStartDirectory }

const NO_FOLDER_KEY = "none"
/** Sessions the hover menu of a row lists, newest first. */
const HOVER_SESSION_LIMIT = 8
/** Time the hover menu stays after the pointer leaves, enough to cross the gap into it. */
const HOVER_CLOSE_MS = 150

/**
 * Directories a chat's first message can start a Pi session in: every checkout under `~/git` and every file mount, the most recently used first, the rest by name.
 *
 * - A checkout counts as used when a session ran in it, below it or in one of its worktrees, a mount when a session ran in it or below it
 */
export function piStartDirectories(repositories: readonly PiRepository[], mounts: readonly string[], sessions: readonly PiSessionInfo[]): PiStartDirectory[] {
  const labels = new Map<string, string>()
  for (const mount of mounts) labels.set(trimSlash(mount), baseName(trimSlash(mount)))
  for (const repository of repositories) labels.set(trimSlash(repository.path), repository.name)
  const latest = new Map<string, PiSessionInfo>()
  for (const session of sessions) {
    const repository = piSessionRepository(session, repositories)
    for (const path of labels.keys()) {
      if (!isInside(session.cwd, path) && trimSlash(repository?.path ?? "") !== path) continue
      const known = latest.get(path)
      if (!known || session.modifiedAt > known.modifiedAt) latest.set(path, session)
    }
  }
  return [...labels.entries()]
    .map(([path, label]) => ({ path, label, latest: latest.get(path) ?? null }))
    .sort((left, right) => (right.latest?.modifiedAt ?? "").localeCompare(left.latest?.modifiedAt ?? "") || PATH_COLLATOR.compare(left.label, right.label))
}

/**
 * Dropdown that picks the directory a chat's first message starts a Pi session in, `None` keeps a plain chat.
 *
 * - Lists the checkouts under `~/git` (`/api/pi/repos`) and the file mounts, the most recently used first by their coding sessions, searched with a fuzzy finder
 * - A row shows how long ago its newest coding session ran, hovering it lists its newest sessions beside the list, a click on one opens it
 * - Picks any listed folder, the caller refuses one Pi cannot start in (a mount outside `~/git`)
 * - Hidden without host access, Pi needs it
 */
export function PiWorkspacePicker({ folder, onChange, onOpenSession }: {
  folder: string | null
  onChange: (folder: string | null) => void
  onOpenSession: (session: PiSessionInfo, repo: string | null) => void
}) {
  const hostAccess = useHostAccess()
  const { sessions, error } = usePiSessions({ profile: "coding" })
  const repositories = usePiRepositories()
  const [mounts, setMounts] = useState<string[]>([])
  const [mountsFailed, setMountsFailed] = useState(false)
  const [query, setQuery] = useState("")
  const [showUnused, setShowUnused] = useState(false)
  const popover = usePillPopover(useCallback(() => setQuery(""), []))
  const { open } = popover
  // Rows age from the latest opening, set when its mount read settles
  const [now, setNow] = useState(Date.now)
  // The row whose sessions the hover menu lists, and its offset from the panel's top
  const [hovered, setHovered] = useState<{ path: string; top: number } | null>(null)
  const closeTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const keepHover = () => clearTimeout(closeTimer.current)
  const leaveHover = () => {
    clearTimeout(closeTimer.current)
    closeTimer.current = setTimeout(() => setHovered(null), HOVER_CLOSE_MS)
  }
  const hoverRow = (path: string | null, row: HTMLElement) => {
    keepHover()
    const panel = popover.panelRef.current
    if (path === null || !panel) return setHovered(null)
    const rowBox = (row.parentElement ?? row).getBoundingClientRect()
    setHovered({ path, top: rowBox.top - panel.getBoundingClientRect().top })
  }

  // Read on every opening, so a mount added meanwhile shows
  useLoadWhile(open, fileRootTree, ({ tree }) => {
    setMounts(tree.map(mountOf))
    setMountsFailed(false)
    setNow(Date.now())
  }, () => {
    setMountsFailed(true)
    setNow(Date.now())
  })

  if (!hostAccess) return null

  const trimmedQuery = query.trim()
  const searching = trimmedQuery.length > 0
  const directories = piStartDirectories(repositories, mounts, sessions)
  const folders = searching ? rankByScore(directories, (directory) => fuzzyPathScore(directory.path, trimmedQuery)) : directories
  // Never-used folders sit behind a chevron row, a search shows them all
  const used = searching ? folders : folders.filter((directory) => directory.latest)
  const unused = searching ? [] : folders.filter((directory) => !directory.latest)
  const asEntries = (list: PiStartDirectory[]): Entry[] => list.map((directory) => ({ kind: "folder", directory }))
  const groups: PickerGroup<Entry>[] = [
    { key: "none", items: searching ? [] : [{ kind: "none" }] },
    { key: "folders", label: searching ? undefined : "Projects", items: asEntries(used) },
    { key: "unused", items: unused.length === 0 ? [] : [{ kind: "unused", count: unused.length }, ...(showUnused ? asEntries(unused) : [])] },
  ]
  const name = folder === null ? null : baseName(folder)
  const hoveredSessions = hovered && open
    ? piSessionsInside(sessions, hovered.path, repositories).sort((left, right) => right.modifiedAt.localeCompare(left.modifiedAt)).slice(0, HOVER_SESSION_LIMIT)
    : []
  const panel = popover.panelRef.current
  // Portaled into the panel, so a press on it counts as inside the popover; clicks stop here, the row under it in React would pick its folder
  const hoverMenu = hovered && panel && hoveredSessions.length > 0 && createPortal(<div
    role="menu"
    aria-label={`Sessions in ${baseName(hovered.path)}`}
    onMouseEnter={keepHover}
    onMouseLeave={leaveHover}
    onClick={(event) => event.stopPropagation()}
    onMouseDown={(event) => event.preventDefault()}
    style={{ top: hovered.top }}
    className="absolute left-full ml-2 flex w-64 flex-col rounded-2xl bg-surface p-1 shadow-[var(--shadow-md)]"
  >
    {hoveredSessions.map((session) => <button
      key={session.sessionId}
      type="button"
      role="menuitem"
      onClick={() => {
        onOpenSession(session, piSessionRepository(session, repositories)?.name ?? null)
        popover.close()
      }}
      className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-xs text-ink-muted transition-colors duration-100 hover:bg-surface-elevated hover:text-ink focus-visible:bg-surface-elevated focus-visible:outline-none"
    >
      <span className="min-w-0 flex-1 truncate">{piSessionTitle(session)}</span>
      <span className="shrink-0 text-[11px] tabular-nums text-ink-faint">{compactElapsed(Date.parse(session.modifiedAt), now)}</span>
    </button>)}
  </div>, panel)
  return <CommandPicker<Entry>
    triggerLabel={`Project: ${name ?? "none"}`}
    triggerContent={<>
      <span className="max-w-32 truncate text-[13px]">{name ?? "None"}</span>
    </>}
    triggerClassName="h-8 px-2 text-ink-faint hover:bg-surface-elevated hover:text-ink-muted"
    label="Project"
    popover={popover}
    groups={groups}
    itemKey={(entry) => (entry.kind === "none" ? NO_FOLDER_KEY : entry.kind === "unused" ? "unused" : entry.directory.path)}
    searchPlaceholder="Search projects…"
    query={query}
    onQueryChange={setQuery}
    initialKey={folder ?? NO_FOLDER_KEY}
    header={mountsFailed ? "Could not list file mounts" : error ?? undefined}
    emptyText="No projects"
    renderItem={(entry) => {
      if (entry.kind === "none") return <span onMouseEnter={(event) => hoverRow(null, event.currentTarget)} className="flex min-w-0 flex-1 items-center gap-2">
        <span className="min-w-0 flex-1 truncate">None</span>
        {folder === null && <Check aria-hidden className="h-3.5 w-3.5 shrink-0 text-ink-muted" />}
      </span>
      if (entry.kind === "unused") return <span onMouseEnter={(event) => hoverRow(null, event.currentTarget)} className="flex min-w-0 flex-1 items-center gap-1.5 text-ink-faint">
        {showUnused ? <ChevronDown aria-hidden className="h-3 w-3 shrink-0" /> : <ChevronRight aria-hidden className="h-3 w-3 shrink-0" />}
        <span className="min-w-0 flex-1 truncate text-[13px] leading-5">Not yet used</span>
        <span className="shrink-0 text-[11px] tabular-nums">{entry.count}</span>
      </span>
      const { path, label, latest } = entry.directory
      const age = latest && compactElapsed(Date.parse(latest.modifiedAt), now)
      return <span onMouseEnter={(event) => hoverRow(path, event.currentTarget)} onMouseLeave={leaveHover} className="flex min-w-0 flex-1 items-center gap-2">
        <span className="min-w-0 flex-1 truncate text-[13px] leading-5">{label}</span>
        {age && <span className="shrink-0 pl-1 text-[11px] tabular-nums text-ink-faint">{age}</span>}
        {path === folder && <Check aria-hidden className="h-3.5 w-3.5 shrink-0 text-ink-muted" />}
        {hovered?.path === path && hoverMenu}
      </span>
    }}
    onPick={(entry) => {
      if (entry.kind === "unused") {
        setShowUnused((shown) => !shown)
        return false
      }
      onChange(entry.kind === "none" ? null : entry.directory.path)
      return true
    }}
    panelClassName="w-72 overflow-visible rounded-2xl border-transparent bg-surface shadow-[var(--shadow-md)]"
  />
}
