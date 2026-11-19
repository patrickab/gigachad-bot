"use client"

import { useCallback, useState } from "react"
import { Check, ChevronDown, Folder } from "lucide-react"
import { CommandPicker, type PickerGroup } from "@/components/ui/CommandPicker"
import { usePillPopover } from "@/components/ui/PillPopover"
import { useHostAccess } from "@/contexts/SettingsContext"
import { useLoadWhile } from "@/hooks/useLoadWhile"
import { fileRootTree, type PiRepository, type PiSessionInfo } from "@/lib/backend/api"
import { mountOf } from "@/lib/domain/files"
import { fuzzyPathScore, rankByScore } from "@/lib/fuzzy"
import { PATH_COLLATOR, baseName, dirName, isInside, shortPath, trimSlash } from "@/lib/paths"
import { cn, compactElapsed } from "@/lib/utils"
import { usePiRepositories, usePiSessions } from "./sessionStore"
import { piSessionRepository, piSessionTitle } from "./usePiSidebar"

/** A directory a Pi session can be asked to start in, with the newest coding session that ran in it or below it. */
export interface PiStartDirectory {
  path: string
  label: string
  latest: PiSessionInfo | null
}

type Entry = { kind: "none" } | { kind: "folder"; directory: PiStartDirectory }

const NO_FOLDER_KEY = "none"

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
 * - A row shows its newest coding session and how long ago it ran
 * - Picks any listed folder, the caller refuses one Pi cannot start in (a mount outside `~/git`)
 * - Hidden without host access, Pi needs it
 */
export function PiWorkspacePicker({ folder, onChange }: { folder: string | null; onChange: (folder: string | null) => void }) {
  const hostAccess = useHostAccess()
  const { sessions, error } = usePiSessions({ profile: "coding" })
  const repositories = usePiRepositories()
  const [mounts, setMounts] = useState<string[]>([])
  const [mountsFailed, setMountsFailed] = useState(false)
  const [query, setQuery] = useState("")
  const popover = usePillPopover(useCallback(() => setQuery(""), []))
  const { open } = popover
  // Rows age from the latest opening, set when its mount read settles
  const [now, setNow] = useState(Date.now)

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
  const groups: PickerGroup<Entry>[] = [
    { key: "none", items: searching ? [] : [{ kind: "none" }] },
    { key: "folders", label: searching ? undefined : "Projects", items: folders.map((directory): Entry => ({ kind: "folder", directory })) },
  ]
  const name = folder === null ? null : baseName(folder)

  return <CommandPicker<Entry>
    triggerLabel={`Project: ${name ?? "none"}`}
    triggerContent={<>
      <span className="max-w-32 truncate">{name ?? "None"}</span>
      <ChevronDown aria-hidden className="h-3 w-3 shrink-0" />
    </>}
    triggerClassName={cn("h-8 px-2 text-ink-faint hover:bg-surface-elevated", name !== null && "text-ink")}
    label="Project"
    popover={popover}
    groups={groups}
    itemKey={(entry) => (entry.kind === "none" ? NO_FOLDER_KEY : entry.directory.path)}
    searchPlaceholder="Search projects…"
    query={query}
    onQueryChange={setQuery}
    initialKey={folder ?? NO_FOLDER_KEY}
    header={mountsFailed ? "Could not list file mounts" : error ?? undefined}
    emptyText="No projects"
    renderItem={(entry) => {
      if (entry.kind === "none") return <>
        <span className="min-w-0 flex-1 truncate">None</span>
        {folder === null && <Check aria-hidden className="h-3.5 w-3.5 shrink-0 text-ink-muted" />}
      </>
      const { path, label, latest } = entry.directory
      const age = latest && compactElapsed(Date.parse(latest.modifiedAt), now)
      return <>
        <Folder aria-hidden className="h-3.5 w-3.5 shrink-0 text-ink-faint" />
        <span className="flex min-w-0 flex-1 flex-col">
          <span className="truncate">{label}</span>
          {searching
            ? <span className="truncate text-[10px] text-ink-faint">{shortPath(dirName(path))}</span>
            : latest && <span className="truncate text-[10px] text-ink-faint">{piSessionTitle(latest)}</span>}
        </span>
        {age && <span className="shrink-0 text-[10px] text-ink-faint">{age}</span>}
        {path === folder && <Check aria-hidden className="h-3.5 w-3.5 shrink-0 text-ink-muted" />}
      </>
    }}
    onPick={(entry) => {
      onChange(entry.kind === "none" ? null : entry.directory.path)
      return true
    }}
    panelClassName="max-h-[45vh] w-72"
  />
}
