"use client"

import { useCallback, useState } from "react"
import { Check, ChevronDown, Folder } from "lucide-react"
import { CommandPicker, type PickerGroup } from "@/components/ui/CommandPicker"
import { usePillPopover } from "@/components/ui/PillPopover"
import { useHostAccess } from "@/contexts/SettingsContext"
import { useLoadWhile } from "@/hooks/useLoadWhile"
import { fileRootTree } from "@/lib/backend/api"
import { mountOf } from "@/lib/domain/files"
import { fuzzyPathScore, rankByScore } from "@/lib/fuzzy"
import { baseName, dirName, shortPath } from "@/lib/paths"
import { cn } from "@/lib/utils"
import { useOmp } from "./OmpProvider"
import { startDirectories } from "./sessions"

type Entry = { kind: "none" } | { kind: "folder"; path: string }

const NO_FOLDER_KEY = "none"

/**
 * Dropdown that picks the directory a chat's first message starts an omp session in, `None` keeps a plain chat.
 *
 * - Lists the file mounts and omp-web workspaces, the most recently used first, searched with a fuzzy finder
 * - Hidden without host access, omp needs it
 */
export function WorkspacePicker({ folder, onChange }: { folder: string | null; onChange: (folder: string | null) => void }) {
  const hostAccess = useHostAccess()
  const { overview } = useOmp()
  const [mounts, setMounts] = useState<string[]>([])
  const [mountsFailed, setMountsFailed] = useState(false)
  const [query, setQuery] = useState("")
  const popover = usePillPopover(useCallback(() => setQuery(""), []))
  const { open } = popover

  // Read on every opening, so a mount added meanwhile shows
  useLoadWhile(open, fileRootTree, ({ tree }) => {
    setMounts(tree.map(mountOf))
    setMountsFailed(false)
  }, () => setMountsFailed(true))

  if (!hostAccess) return null

  const trimmedQuery = query.trim()
  const searching = trimmedQuery.length > 0
  const directories = startDirectories(mounts, overview)
  const folders = searching ? rankByScore(directories, (path) => fuzzyPathScore(path, trimmedQuery)) : directories
  const groups: PickerGroup<Entry>[] = [
    { key: "none", items: searching ? [] : [{ kind: "none" }] },
    { key: "folders", label: searching ? undefined : "Projects", items: folders.map((path): Entry => ({ kind: "folder", path })) },
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
    itemKey={(entry) => (entry.kind === "none" ? NO_FOLDER_KEY : entry.path)}
    searchPlaceholder="Search projects…"
    query={query}
    onQueryChange={setQuery}
    initialKey={folder ?? NO_FOLDER_KEY}
    header={mountsFailed ? "Could not list file mounts" : overview ? undefined : "omp-web is not reachable"}
    emptyText="No projects"
    renderItem={(entry) => {
      const selected = entry.kind === "none" ? folder === null : entry.path === folder
      return <>
        {entry.kind === "folder" && <Folder aria-hidden className="h-3.5 w-3.5 shrink-0 text-ink-faint" />}
        <span className="min-w-0 flex-1 truncate">{entry.kind === "none" ? "None" : baseName(entry.path)}</span>
        {entry.kind === "folder" && searching && <span className="max-w-[45%] truncate text-[10px] text-ink-faint">{shortPath(dirName(entry.path))}</span>}
        {selected && <Check aria-hidden className="h-3.5 w-3.5 shrink-0 text-ink-muted" />}
      </>
    }}
    onPick={(entry) => {
      onChange(entry.kind === "none" ? null : entry.path)
      return true
    }}
    panelClassName="max-h-[45vh] w-72"
  />
}
