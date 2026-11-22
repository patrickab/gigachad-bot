"use client"

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react"
import { FloatingPicker, isRemoveRowKey, neighbourKey, type PickerGroup } from "@/components/ui/CommandPicker"
import { ErrorText } from "@/components/ui/ErrorText"
import { useHostAccess } from "@/contexts/SettingsContext"
import { useTabActive, type WorkspaceItem } from "@/contexts/TabContext"
import { useEventListener } from "@/hooks/useEventListener"
import { useResetOnChange } from "@/hooks/useResetOnChange"
import { archivePiSession, type PiRepository, type PiSessionInfo } from "@/lib/backend/api"
import { abbreviatePath, baseName, shortPath } from "@/lib/paths"
import { cn, compactElapsed, errorMessage, isAltOnly, isTerminalKeyEvent } from "@/lib/utils"
import { refreshPiSessions, searchPiSessions, usePiRepositories, usePiSessions, type PiSearchHit } from "./sessionStore"
import { PiSessionPreview, prefetchPiSessionPreviews } from "./PiSessionPreview"
import { PiSessionDot, piTokenText, piSessionError, piSessionItem, piSessionRepository, piSessionTitle, piSessionWorktree } from "./usePiSidebar"

const TICK_MS = 60_000
/** Typing pauses this long before the transcripts are searched. */
const SEARCH_DELAY_MS = 200
/** Widest a row's title and its details line get, one value so both edges line up. */
const ROW_TEXT_WIDTH = "max-w-[16rem]"

interface Entry { session: PiSessionInfo; snippet?: string }

/** Newest first by last use. */
function newestFirst(sessions: readonly PiSessionInfo[]): PiSessionInfo[] {
  return [...sessions].sort((left, right) => right.modifiedAt.localeCompare(left.modifiedAt))
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

/** `path` whole while it fits its box, else abbreviated to its first and last dir. */
function FittedPath({ path }: { path: string }) {
  const ref = useRef<HTMLSpanElement>(null)
  const [short, setShort] = useState(false)
  useLayoutEffect(() => {
    const element = ref.current
    if (!element) return
    const fit = () => {
      element.textContent = path
      const overflows = element.scrollWidth > element.clientWidth
      element.textContent = overflows ? abbreviatePath(path) : path
      setShort(overflows)
    }
    fit()
    if (typeof ResizeObserver === "undefined") return
    const observer = new ResizeObserver(fit)
    observer.observe(element)
    return () => observer.disconnect()
  }, [path])
  return <span ref={ref} className="min-w-0 flex-1 truncate">{short ? abbreviatePath(path) : path}</span>
}

function SessionRow({ session, snippet, repositories, now }: { session: PiSessionInfo; snippet?: string; repositories: readonly PiRepository[]; now: number }) {
  const worktree = piSessionWorktree(session, piSessionRepository(session, repositories))
  return <span className="flex min-w-0 flex-1 flex-col py-0.5">
    <span className="flex min-w-0 items-center gap-2">
      <span className="flex w-1.5 shrink-0 justify-center"><PiSessionDot sessionId={session.sessionId} /></span>
      <span className={cn("truncate leading-5", ROW_TEXT_WIDTH)}>{piSessionTitle(session)}</span>
    </span>
    {snippet
      ? <span className={cn("ml-3.5 truncate text-[10px] text-ink-subtle", ROW_TEXT_WIDTH)}>{snippet.replace(/\s+/g, " ").trim()}</span>
      : <span className={cn("ml-3.5 flex min-w-0 gap-2 text-[10px] tabular-nums text-ink-subtle", ROW_TEXT_WIDTH)}>
        <span className="w-6 shrink-0">{compactElapsed(Date.parse(session.modifiedAt), now)}</span>
        <span className="w-8 shrink-0 text-right">{piTokenText(session.tokens)}</span>
        <FittedPath path={shortPath(session.cwd)} />
        {worktree && <span className="max-w-[6rem] shrink-0 truncate">{worktree}</span>}
      </span>}
  </span>
}

/**
 * Panel over every Pi coding session, opened by Alt+A (G5) and docked under the top bar, the most recently used first; a hovered session previews its transcript.
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
  const [archiving, setArchiving] = useState<ReadonlySet<string>>(() => new Set())
  const [archiveError, setArchiveError] = useState("")
  const search = useSearchHits(query)
  const [open, setOpen] = useState(false)
  const close = useCallback(() => { setOpen(false); setQuery("") }, [])
  const tabActive = useTabActive()
  const sessionList = useMemo(() => newestFirst(listed), [listed])
  // The recent sessions' previews are read ahead after load and whenever the list changes, so hovering shows them at once
  useEffect(() => (hostAccess ? prefetchPiSessionPreviews(sessionList) : undefined), [hostAccess, sessionList])
  // Alt+A toggles the picker from anywhere in the tab, terminals keep their keys
  useEventListener("window", "keydown", (event) => {
    if (!isAltOnly(event) || event.key.toLowerCase() !== "a" || isTerminalKeyEvent(event)) return
    event.preventDefault()
    if (open) close()
    else setOpen(true)
  }, { active: tabActive && Boolean(hostAccess), capture: true })
  // The list lies in a portal, a tab sent to the background takes its open list along
  useResetOnChange(tabActive, (active) => { if (!active) close() })
  if (!hostAccess || !open) return null

  const sessions = sessionList.filter((session) => !archiving.has(session.sessionId))
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
      { key: "named", label: "Sessions", items: named.map((session) => ({ session })) },
      { key: "content", label: "Content", items: hits.map((hit) => ({ session: hit.session, snippet: hit.snippet })) },
    )
  } else {
    groups.push({ key: "all", items: sessions.map((session) => ({ session })) })
  }
  const error = archiveError || search.error || listError

  return <FloatingPicker<Entry>
    label="Agents"
    onClose={close}
    groups={groups}
    itemKey={(entry) => (entry.session.sessionId)}
    searchPlaceholder="Search sessions…"
    query={query}
    onQueryChange={setQuery}
    header={error ? <ErrorText className="text-[10px]">{error}</ErrorText> : undefined}
    emptyText={words.length > 0 && search.hits === null ? "Searching…" : "No sessions"}
    renderItem={(entry) => <SessionRow session={entry.session} snippet={entry.snippet} repositories={repositories} now={now} />}
    renderPreview={(entry) => <PiSessionPreview session={entry.session} />}
    onKeyDown={(event, entry, list) => {
      if (!isRemoveRowKey(event) || !entry) return false
      event.preventDefault()
      const neighbour = neighbourKey(list, entry.session.sessionId)
      if (neighbour) list.highlight(neighbour)
      void archive(entry.session)
      return true
    }}
    onPick={(entry) => {
      const { session } = entry
      onOpenItem(piSessionItem(session, piSessionRepository(session, repositories)?.name ?? baseName(session.cwd), null))
      return true
    }}
  />
}
