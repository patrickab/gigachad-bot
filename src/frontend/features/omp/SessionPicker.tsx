"use client"

import { useEffect, useRef, useState } from "react"
import { Bot } from "lucide-react"
import { CommandPicker, FINDER_INSET_VAR, isRemoveRowKey, neighbourKey, type PickerGroup } from "@/components/ui/CommandPicker"
import { ErrorText } from "@/components/ui/ErrorText"
import { usePillPopover } from "@/components/ui/PillPopover"
import { StatusDot } from "@/components/ui/StatusDot"
import { useHostAccess } from "@/contexts/SettingsContext"
import { useTabActive, type WorkspaceItem } from "@/contexts/TabContext"
import { useEventListener } from "@/hooks/useEventListener"
import { ChatPair } from "@/features/chat/ChatPair"
import { deleteOmpSession, fetchOmpSessionHistory } from "@/lib/backend/api"
import type { Message, OmpSession } from "@/lib/domain/types"
import { shortPath } from "@/lib/paths"
import { cn, compactElapsed, errorMessage, isAltOnly, isTerminalKeyEvent, toggled } from "@/lib/utils"
import { useOmp } from "./OmpProvider"
import { groupSessions, historyPairs, sessionDirectory, sessionTitle } from "./sessions"
import { ompItem } from "./useOmpSidebar"

const TICK_MS = 60_000
/** Arrowing through rows fast changes the highlight on every row, only a resting one fetches. */
const PREVIEW_DELAY_MS = 120
/** Widest a row's title and its details line get, one value so both edges line up. It sets the finder's width. */
const ROW_TEXT_WIDTH = "max-w-[13rem]"

type HistoryMessage = Pick<Message, "role" | "content">

type Entry = { kind: "session"; session: OmpSession; running: boolean } | { kind: "older"; count: number }

/** Current time in ms, renewed every minute so rows age without a refetch. */
function useNowMs(): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), TICK_MS)
    return () => clearInterval(timer)
  }, [])
  return now
}

/** The highlighted session as Q&A pairs like a chat, the last one expanded, fetched once per session version into `cache`. */
function SessionPreview({ session, cache }: { session: OmpSession; cache: Map<string, HistoryMessage[]> }) {
  const version = `${session.id}@${session.modified}`
  const [loaded, setLoaded] = useState<{ version: string; messages: HistoryMessage[] | null; error: string | null } | null>(null)
  const [overrides, setOverrides] = useState<ReadonlyMap<number, boolean>>(() => new Map())
  const scroller = useRef<HTMLDivElement>(null)
  useEffect(() => {
    let cancelled = false
    const show = (messages: HistoryMessage[] | null, error: string | null) => { if (!cancelled) setLoaded({ version, messages, error }) }
    const timer = setTimeout(() => {
      const cached = cache.get(version)
      if (cached) return show(cached, null)
      fetchOmpSessionHistory(session.id)
        .then((messages) => { cache.set(version, messages); show(messages, null) })
        .catch((reason: unknown) => show(null, errorMessage(reason, "Could not load this session")))
    }, PREVIEW_DELAY_MS)
    return () => { cancelled = true; clearTimeout(timer) }
  }, [version, session.id, cache])
  const current = loaded?.version === version ? loaded : null
  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight })
  }, [current])

  const pairs = current?.messages ? historyPairs(current.messages) : []
  const lastIndex = pairs.at(-1)?.globalIndex
  const isExpanded = (globalIndex: number) => overrides.get(globalIndex) ?? globalIndex === lastIndex
  const toggle = (globalIndex: number) => setOverrides((previous) => new Map(previous).set(globalIndex, !isExpanded(globalIndex)))
  const status = current === null ? "Loading preview…" : current.error ?? (pairs.length === 0 ? "No messages yet" : null)

  // The list docks over the preview's top left corner, the transcript runs beside it
  return <div ref={scroller} style={{ paddingLeft: `var(${FINDER_INSET_VAR})` }} className="h-full overflow-y-auto bg-paper text-ink scrollbar-thin">
    {status ? <p className="px-4 py-3 text-xs text-ink-subtle">{status}</p>
      : <div className="py-3 pr-3">
        {pairs.map(({ user, assistant, globalIndex }) => <ChatPair key={globalIndex}
          user={user}
          assistant={assistant}
          globalIndex={globalIndex}
          expanded={isExpanded(globalIndex)}
          focused={false}
          isStreaming={false}
          onExpand={() => toggle(globalIndex)}
          onCollapse={() => toggle(globalIndex)}
          collapsedRef={() => {}}
        />)}
      </div>}
  </div>
}

function SessionRow({ session, running, now }: { session: OmpSession; running: boolean; now: number }) {
  const age = compactElapsed(Date.parse(session.modified), now)
  const details = [shortPath(session.cwd), session.worktree_branch, age === "now" ? age : `${age} ago`].filter(Boolean)
  return <span className="flex min-w-0 flex-1 flex-col py-0.5">
    <span className="flex min-w-0 items-center gap-2">
      <StatusDot tone={running ? "active" : "idle"} />
      <span className={cn("truncate leading-5", ROW_TEXT_WIDTH)}>{sessionTitle(session)}</span>
    </span>
    <span className={cn("ml-3.5 truncate text-[10px] text-ink-faint", ROW_TEXT_WIDTH)}>{details.join(" · ")}</span>
  </span>
}

/** Top bar panel over every omp-web session: running ones first, then recent, older ones behind "Show N older". Picking one opens it in the current tab, Delete at the search's end deletes the highlighted one. */
export function SessionPicker({ onOpenItem }: { onOpenItem: (item: WorkspaceItem, inNewTab?: boolean) => void }) {
  const hostAccess = useHostAccess()
  const { overview, refreshOverview } = useOmp()
  const now = useNowMs()
  const [showOlder, setShowOlder] = useState(false)
  const [deleting, setDeleting] = useState<ReadonlySet<string>>(() => new Set())
  const [deleteError, setDeleteError] = useState("")
  const [historyCache] = useState(() => new Map<string, HistoryMessage[]>())
  const popover = usePillPopover()
  const tabActive = useTabActive()
  // Alt+A toggles the picker from anywhere in the tab, terminals keep their keys
  useEventListener("window", "keydown", (event) => {
    if (!isAltOnly(event) || event.key.toLowerCase() !== "a" || isTerminalKeyEvent(event)) return
    event.preventDefault()
    if (popover.open) popover.close()
    else popover.show()
  }, { active: tabActive && Boolean(hostAccess), capture: true })
  if (!hostAccess) return null

  const sessions = (overview?.sessions ?? []).filter((session) => !deleting.has(session.id))
  const { running, recent, older } = groupSessions(sessions, overview?.running_session_ids ?? [], now)
  const deleteSession = async (session: OmpSession) => {
    setDeleteError("")
    setDeleting((current) => toggled(current, session.id))
    try {
      await deleteOmpSession(session.id)
      refreshOverview()
    } catch (cause) {
      setDeleting((current) => toggled(current, session.id))
      setDeleteError(errorMessage(cause, "Could not delete this session"))
    }
  }

  const asEntries = (list: OmpSession[], isRunning: boolean): Entry[] => list.map((session) => ({ kind: "session", session, running: isRunning }))
  const groups: PickerGroup<Entry>[] = [
    { key: "running", label: "Running", items: asEntries(running, true) },
    { key: "recent", label: "Recent", items: asEntries(recent, false) },
  ]
  if (older.length > 0) {
    groups.push(showOlder
      ? { key: "older", label: "Older", items: asEntries(older, false) }
      : { key: "older", items: [{ kind: "older", count: older.length }] })
  }
  const runningCount = running.length

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
    itemKey={(entry) => (entry.kind === "session" ? entry.session.id : "older")}
    searchPlaceholder="Search sessions…"
    searchText={(entry) => entry.kind === "session"
      ? [sessionTitle(entry.session), entry.session.cwd, entry.session.first_message].join(" ")
      : ""}
    header={deleteError ? <ErrorText className="text-[10px]">{deleteError}</ErrorText> : overview ? undefined : "omp-web is not reachable"}
    emptyText="No sessions"
    renderItem={(entry) => entry.kind === "session"
      ? <SessionRow session={entry.session} running={entry.running} now={now} />
      : <span className="py-0.5 text-[11px] text-ink-subtle">Show {entry.count} older</span>}
    renderPreview={(entry) => (entry.kind === "session" ? <SessionPreview key={entry.session.id} session={entry.session} cache={historyCache} /> : null)}
    onKeyDown={(event, entry, list) => {
      if (!isRemoveRowKey(event) || entry?.kind !== "session") return false
      event.preventDefault()
      const neighbour = neighbourKey(list, entry.session.id)
      if (neighbour) list.highlight(neighbour)
      void deleteSession(entry.session)
      return true
    }}
    onPick={(entry) => {
      if (entry.kind === "older") {
        setShowOlder(true)
        return false
      }
      const { session } = entry
      onOpenItem(ompItem({ kind: "session", sessionId: session.id }, sessionDirectory(session), null))
      return true
    }}
    placement="below"
  />
}
