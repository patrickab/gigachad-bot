"use client"

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react"
import { FloatingPicker, isRemoveRowKey, neighbourKey, useAltKeyPicker, type PickerGroup } from "@/components/ui/CommandPicker"
import { ErrorText } from "@/components/ui/ErrorText"
import { useHostAccess } from "@/contexts/SettingsContext"
import type { WorkspaceItem } from "@/contexts/TabContext"
import { useDebouncedSearch } from "@/hooks/useDebouncedSearch"
import { useNowMs } from "@/hooks/useNowMs"
import { useResetOnChange } from "@/hooks/useResetOnChange"
import { archivePiSession, type PiRepository, type PiSessionInfo, type PiSessionMatch } from "@/lib/backend/api"
import { abbreviatePath, baseName, shortPath } from "@/lib/paths"
import { cn, compactElapsed } from "@/lib/utils"
import { focusPiSearchHit } from "./searchFocus"
import { refreshPiSessions, searchPiSessions, usePiRepositories, usePiSessions } from "./sessionStore"
import { PiSessionPreview, prefetchPiSessionPreviews } from "./PiSessionPreview"
import { PiSessionDot, piTokenText, piSessionError, piSessionItem, piSessionRepository, piSessionTitle, piSessionWorktree } from "./usePiSidebar"

/** Widest a row's title and its details line get, one value so both edges line up. */
const ROW_TEXT_WIDTH = "max-w-[16rem]"

/** A row: a session, or one hit of a session found by its content (`listed`: the session's hits the search lists, `first` its first row, the one that names the session). */
interface Entry { session: PiSessionInfo; snippet?: string; hit?: PiSessionMatch; listed?: readonly PiSessionMatch[]; first?: boolean }

const entryKey = ({ session, hit }: Entry) => (hit ? `${session.sessionId}:${hit.entryId}:${hit.ordinal}` : session.sessionId)

/** Newest first by last use. */
function newestFirst(sessions: readonly PiSessionInfo[]): PiSessionInfo[] {
  return [...sessions].sort((left, right) => right.modifiedAt.localeCompare(left.modifiedAt))
}

const searchCodingSessions = (query: string, signal: AbortSignal) => searchPiSessions(query, { profile: "coding" }, signal)

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

/** The text around a hit, the matched part in full ink. */
function HitText({ hit }: { hit: PiSessionMatch }) {
  return <>{hit.before}<span className="text-ink">{hit.match}</span>{hit.after}</>
}

function SessionRow({ entry: { session, snippet, hit, first = true }, repositories, now }: { entry: Entry; repositories: readonly PiRepository[]; now: number }) {
  const worktree = piSessionWorktree(session, piSessionRepository(session, repositories))
  if (hit && !first) {
    return <span className={cn("ml-3.5 truncate py-0.5 text-[10px] text-ink-subtle", ROW_TEXT_WIDTH)}><HitText hit={hit} /></span>
  }
  return <span className="flex min-w-0 flex-1 flex-col py-0.5">
    <span className="flex min-w-0 items-center gap-2">
      <span className="flex w-1.5 shrink-0 justify-center"><PiSessionDot sessionId={session.sessionId} /></span>
      <span className={cn("truncate leading-5", ROW_TEXT_WIDTH)}>{piSessionTitle(session)}</span>
    </span>
    {hit || snippet
      ? <span className={cn("ml-3.5 truncate text-[10px] text-ink-subtle", ROW_TEXT_WIDTH)}>{hit ? <HitText hit={hit} /> : snippet!.replace(/\s+/g, " ").trim()}</span>
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
 * - A content match lists the session's best hits, picking one opens the session at that hit (`focusPiSearchHit`)
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
  const search = useDebouncedSearch(query.trim(), searchCodingSessions)
  const { open, close } = useAltKeyPicker("a", hostAccess)
  useResetOnChange(open, (isOpen) => { if (!isOpen) setQuery("") })
  const sessionList = useMemo(() => newestFirst(listed), [listed])
  // The recent sessions' previews are read ahead after load and whenever the list changes, so hovering shows them at once
  useEffect(() => (hostAccess ? prefetchPiSessionPreviews(sessionList) : undefined), [hostAccess, sessionList])
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
      { key: "content", label: "Content", items: hits.flatMap((hit): Entry[] => hit.matches.length > 0
        ? hit.matches.map((match, index) => ({ session: hit.session, hit: match, listed: hit.matches, first: index === 0 }))
        : [{ session: hit.session, snippet: hit.snippet }]) },
    )
  } else {
    groups.push({ key: "all", items: sessions.map((session) => ({ session })) })
  }
  const error = archiveError || search.error || listError

  return <FloatingPicker<Entry>
    label="Agents"
    onClose={close}
    groups={groups}
    itemKey={entryKey}
    searchPlaceholder="Search sessions…"
    query={query}
    onQueryChange={setQuery}
    header={error ? <ErrorText className="text-[10px]">{error}</ErrorText> : undefined}
    emptyText={words.length > 0 && search.hits === null ? "Searching…" : "No sessions"}
    renderItem={(entry) => <SessionRow entry={entry} repositories={repositories} now={now} />}
    renderPreview={(entry) => <PiSessionPreview session={entry.session} />}
    onKeyDown={(event, entry, list) => {
      if (!isRemoveRowKey(event) || !entry) return false
      event.preventDefault()
      const neighbour = neighbourKey(list, entryKey(entry))
      if (neighbour) list.highlight(neighbour)
      void archive(entry.session)
      return true
    }}
    onPick={(entry) => {
      const { session, hit } = entry
      if (hit) focusPiSearchHit(session.sessionId, query.trim(), hit, entry.listed)
      onOpenItem(piSessionItem(session, piSessionRepository(session, repositories)?.name ?? baseName(session.cwd), null))
      return true
    }}
  />
}
