"use client"

import { useCallback, useEffect, useState } from "react"
import { Bot } from "lucide-react"
import { CommandPicker, isRemoveRowKey, neighbourKey, type PickerGroup } from "@/components/ui/CommandPicker"
import { ErrorText } from "@/components/ui/ErrorText"
import { usePillPopover } from "@/components/ui/PillPopover"
import { useHostAccess } from "@/contexts/SettingsContext"
import { useTabActive, type WorkspaceItem } from "@/contexts/TabContext"
import { useEventListener } from "@/hooks/useEventListener"
import { archivePiSession, type PiRepository, type PiSessionInfo } from "@/lib/backend/api"
import { baseName, shortPath } from "@/lib/paths"
import { cn, compactElapsed, errorMessage, isAltOnly, isTerminalKeyEvent } from "@/lib/utils"
import { refreshPiSessions, searchPiSessions, usePiRepositories, usePiSessions, type PiSearchHit } from "./sessionStore"
import { PiSessionDot, piSessionError, piSessionItem, piSessionRepository, piSessionTitle, piSessionWorktree } from "./usePiSidebar"

const TICK_MS = 60_000
/** Typing pauses this long before the transcripts are searched. */
const SEARCH_DELAY_MS = 200
/** How long a session that stopped running stays in the "Recent" group. */
export const RECENT_SESSION_MS = 24 * 60 * 60 * 1000
/** Widest a row's title and its details line get, one value so both edges line up. It sets the finder's width. */
const ROW_TEXT_WIDTH = "max-w-[16rem]"

type Entry =
  | { kind: "session"; session: PiSessionInfo; snippet?: string }
  | { kind: "older"; count: number }

export interface PiSessionGroups {
  running: PiSessionInfo[]
  recent: PiSessionInfo[]
  older: PiSessionInfo[]
}

/** `sessions` split into running ones, the rest modified within `RECENT_SESSION_MS` of `nowMs`, and the older rest. Each newest first. */
export function groupPiSessions(sessions: readonly PiSessionInfo[], nowMs: number): PiSessionGroups {
  const groups: PiSessionGroups = { running: [], recent: [], older: [] }
  for (const session of [...sessions].sort((left, right) => right.modifiedAt.localeCompare(left.modifiedAt))) {
    if (session.live?.running || session.live?.waiting) groups.running.push(session)
    else if (nowMs - Date.parse(session.modifiedAt) <= RECENT_SESSION_MS) groups.recent.push(session)
    else groups.older.push(session)
  }
  return groups
}

/** Current time in ms, renewed every minute so rows age without a refetch. */
function useNowMs(): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), TICK_MS)
    return () => clearInterval(timer)
  }, [])
  return now
}

/** Full-text hits of `query` (M24), read once typing pauses, null while they are read. */
function useSearchHits(query: string): { hits: PiSearchHit[] | null; error: string } {
  const [found, setFound] = useState<{ query: string; hits: PiSearchHit[]; error: string } | null>(null)
  const trimmed = query.trim()
  useEffect(() => {
    if (!trimmed) return
    let cancelled = false
    const timer = setTimeout(() => {
      searchPiSessions(trimmed, { profile: "coding" }).then(
        (hits) => { if (!cancelled) setFound({ query: trimmed, hits, error: "" }) },
        (cause: unknown) => { if (!cancelled) setFound({ query: trimmed, hits: [], error: errorMessage(cause, "Search failed") }) },
      )
    }, SEARCH_DELAY_MS)
    return () => { cancelled = true; clearTimeout(timer) }
  }, [trimmed])
  return found?.query === trimmed ? { hits: found.hits, error: found.error } : { hits: null, error: "" }
}

function SessionRow({ session, snippet, repositories, now }: { session: PiSessionInfo; snippet?: string; repositories: readonly PiRepository[]; now: number }) {
  const age = compactElapsed(Date.parse(session.modifiedAt), now)
  const details = snippet
    ? [snippet.replace(/\s+/g, " ").trim()]
    : [shortPath(session.cwd), piSessionWorktree(session, piSessionRepository(session, repositories)), age === "now" ? age : `${age} ago`].filter(Boolean)
  return <span className="flex min-w-0 flex-1 flex-col py-0.5">
    <span className="flex min-w-0 items-center gap-2">
      <span className="flex w-1.5 shrink-0 justify-center"><PiSessionDot sessionId={session.sessionId} /></span>
      <span className={cn("truncate leading-5", ROW_TEXT_WIDTH)}>{piSessionTitle(session)}</span>
    </span>
    <span className={cn("ml-3.5 truncate text-[10px] text-ink-faint", ROW_TEXT_WIDTH)}>{details.join(" · ")}</span>
  </span>
}

/**
 * Top bar panel over every Pi coding session (Alt+A, G5): running ones first, then recent, older ones behind "Show N older".
 *
 * - Typing filters by title and directory and searches the transcripts, matches show the text they matched
 * - Picking a session opens it, or switches to the tab already showing it; Delete at the search's end archives the highlighted one
 */
export function PiSessionPicker({ onOpenItem }: { onOpenItem: (item: WorkspaceItem, inNewTab?: boolean) => void }) {
  const hostAccess = useHostAccess()
  const { sessions: listed, error: listError } = usePiSessions({ profile: "coding" })
  const repositories = usePiRepositories()
  const now = useNowMs()
  const [query, setQuery] = useState("")
  const [showOlder, setShowOlder] = useState(false)
  const [archiving, setArchiving] = useState<ReadonlySet<string>>(() => new Set())
  const [archiveError, setArchiveError] = useState("")
  const search = useSearchHits(query)
  const popover = usePillPopover(useCallback(() => setQuery(""), []))
  const tabActive = useTabActive()
  // Alt+A toggles the picker from anywhere in the tab, terminals keep their keys
  useEventListener("window", "keydown", (event) => {
    if (!isAltOnly(event) || event.key.toLowerCase() !== "a" || isTerminalKeyEvent(event)) return
    event.preventDefault()
    if (popover.open) popover.close()
    else popover.show()
  }, { active: tabActive && Boolean(hostAccess), capture: true })
  if (!hostAccess) return null

  const sessions = listed.filter((session) => !archiving.has(session.sessionId))
  const archive = async (session: PiSessionInfo) => {
    setArchiveError("")
    setArchiving((current) => new Set(current).add(session.sessionId))
    try {
      await archivePiSession(session.sessionId)
      refreshPiSessions()
    } catch (cause) {
      setArchiving((current) => new Set([...current].filter((id) => id !== session.sessionId)))
      setArchiveError(piSessionError(cause, "Could not archive this session"))
    }
  }

  const asEntries = (list: PiSessionInfo[]): Entry[] => list.map((session) => ({ kind: "session", session }))
  const { running, recent, older } = groupPiSessions(sessions, now)
  const words = query.toLowerCase().split(/\s+/).filter(Boolean)
  const groups: PickerGroup<Entry>[] = []
  if (words.length > 0) {
    const named = sessions.filter((session) => {
      const text = [piSessionTitle(session), session.cwd].join(" ").toLowerCase()
      return words.every((word) => text.includes(word))
    })
    const shown = new Set(named.map((session) => session.sessionId))
    const hits = (search.hits ?? []).filter((hit) => !shown.has(hit.session.sessionId) && !archiving.has(hit.session.sessionId))
    groups.push(
      { key: "named", label: "Sessions", items: asEntries(named) },
      { key: "content", label: "Content", items: hits.map((hit) => ({ kind: "session", session: hit.session, snippet: hit.snippet })) },
    )
  } else {
    groups.push({ key: "running", label: "Running", items: asEntries(running) }, { key: "recent", label: "Recent", items: asEntries(recent) })
    if (older.length > 0) {
      groups.push(showOlder
        ? { key: "older", label: "Older", items: asEntries(older) }
        : { key: "older", items: [{ kind: "older", count: older.length }] })
    }
  }
  const runningCount = running.length
  const error = archiveError || search.error || listError

  return <CommandPicker<Entry>
    triggerLabel={`Agents, ${runningCount} running`}
    triggerContent={<span className="relative">
      <Bot aria-hidden className="block h-4 w-4" />
      {runningCount > 0 && <span aria-hidden className="absolute -right-0.5 -top-0.5 h-1.5 w-1.5 rounded-full bg-ink" />}
    </span>}
    triggerClassName={cn("h-7 w-7 justify-center rounded-md px-0 hover:text-ink", runningCount > 0 ? "text-ink" : "text-ink-subtle")}
    label="Agents"
    popover={popover}
    groups={groups}
    itemKey={(entry) => (entry.kind === "session" ? entry.session.sessionId : "older")}
    searchPlaceholder="Search sessions…"
    query={query}
    onQueryChange={setQuery}
    header={error ? <ErrorText className="text-[10px]">{error}</ErrorText> : undefined}
    emptyText={words.length > 0 && search.hits === null ? "Searching…" : "No sessions"}
    renderItem={(entry) => entry.kind === "session"
      ? <SessionRow session={entry.session} snippet={entry.snippet} repositories={repositories} now={now} />
      : <span className="py-0.5 text-[11px] text-ink-subtle">Show {entry.count} older</span>}
    onKeyDown={(event, entry, list) => {
      if (!isRemoveRowKey(event) || entry?.kind !== "session") return false
      event.preventDefault()
      const neighbour = neighbourKey(list, entry.session.sessionId)
      if (neighbour) list.highlight(neighbour)
      void archive(entry.session)
      return true
    }}
    onPick={(entry) => {
      if (entry.kind === "older") {
        setShowOlder(true)
        return false
      }
      const { session } = entry
      onOpenItem(piSessionItem(session, piSessionRepository(session, repositories)?.name ?? baseName(session.cwd), null))
      return true
    }}
    placement="below"
  />
}
