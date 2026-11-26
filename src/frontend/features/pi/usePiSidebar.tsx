"use client"

import { useEffect, useMemo, useState } from "react"
import { Bot, Pin, PinOff } from "lucide-react"
import type { FileTreeItem } from "@/components/ui/file-tree/FileTree"
import { StatusDot, type StatusTone } from "@/components/ui/StatusDot"
import { itemOf, newPiItem, useTab, type WorkspaceItem } from "@/contexts/TabContext"
import { ApiError, deletePiSession, pinPiSession, type PiRepository, type PiSessionInfo } from "@/lib/backend/api"
import { PATH_COLLATOR, baseName, dirName, isInside, relativeIn, trimSlash } from "@/lib/paths"
import { refreshPiSessions, usePiRepositories, usePiSessions, usePiSessionsSummary, usePiSessionStatus, type PiSessionStatus } from "./sessionStore"

/** Session rows a Sessions folder shows before "Show more". */
const SESSION_LIMIT = 5

/** Sidebar, picker and tab title of a session: the first line of its first prompt. */
export function piSessionTitle(session: Pick<PiSessionInfo, "firstMessage">): string {
  return session.firstMessage.trim().split("\n")[0] || "Untitled session"
}

/** pi-host's error codes for archive, restore and pin, as the error line says them. */
const SESSION_ERRORS: Record<string, string> = {
  live: "Session is running",
  not_found: "Session no longer exists",
  already_archived: "Session is already archived",
  not_archived: "Session is not archived",
  exists: "A session file of that name is already there",
}

/** Readable text of a failed session action: a known pi-host error code as a short sentence, else `fallback`. */
export function piSessionError(cause: unknown, fallback: string): string {
  return cause instanceof ApiError && Object.hasOwn(SESSION_ERRORS, cause.message) ? SESSION_ERRORS[cause.message]! : fallback
}

/** Where the backend puts the worktrees of the checkout at `repositoryPath`: `<git home>/worktrees/<repo>/<branch>`. */
function worktreeHome(repositoryPath: string): string {
  return `${dirName(trimSlash(repositoryPath))}/worktrees/${baseName(trimSlash(repositoryPath))}`
}

/** The checkout a session ran in: the one holding its cwd, or whose worktree holds it. */
export function piSessionRepository(session: Pick<PiSessionInfo, "cwd">, repositories: readonly PiRepository[]): PiRepository | undefined {
  return repositories.find((repository) => isInside(session.cwd, repository.path) || isInside(session.cwd, worktreeHome(repository.path)))
}

/** Branch folder of a session that ran in a worktree, else null. */
export function piSessionWorktree(session: Pick<PiSessionInfo, "cwd">, repository: PiRepository | undefined): string | null {
  if (!repository || !isInside(session.cwd, worktreeHome(repository.path))) return null
  return relativeIn(worktreeHome(repository.path), session.cwd).split("/")[0] || null
}

/** The workspace item opening `session` in a pi tab of checkout `repo`. */
export function piSessionItem(session: Pick<PiSessionInfo, "sessionId">, repo: string, projectSlug: string | null): WorkspaceItem {
  return { kind: "pi", path: repo, projectSlug, pi: { sessionId: session.sessionId, repo } }
}

export interface PiSessionNode {
  session: PiSessionInfo
  forks: PiSessionNode[]
}

const byNewest = (left: PiSessionInfo, right: PiSessionInfo) => right.modifiedAt.localeCompare(left.modifiedAt)

/** Forks nested under the session they branched from, a fork whose parent is not in `sessions` is a root. Newest first. */
export function piSessionForest(sessions: readonly PiSessionInfo[]): PiSessionNode[] {
  const nodes = new Map(sessions.map((session) => [session.sessionPath, { session, forks: [] as PiSessionNode[] }]))
  const roots: PiSessionNode[] = []
  for (const session of [...sessions].sort(byNewest)) {
    const node = nodes.get(session.sessionPath)!
    const parent = session.parentSession ? nodes.get(session.parentSession) : undefined
    if (parent && parent !== node) parent.forks.push(node)
    else roots.push(node)
  }
  return roots
}

/** Sessions belonging to `directory`: those whose checkout (worktrees under their main checkout) lies inside it. */
export function piSessionsInside(sessions: readonly PiSessionInfo[], directory: string, repositories: readonly PiRepository[]): PiSessionInfo[] {
  return sessions.filter((session) => isInside(trimSlash(piSessionRepository(session, repositories)?.path ?? session.cwd), directory))
}

export interface PiWorkspaceGroup {
  path: string
  label: string
  sessions: PiSessionInfo[]
}

/** Checkouts holding sessions outside every mount, by name. Sessions outside every checkout cannot reopen and are left out. */
export function unmountedPiWorkspaces(sessions: readonly PiSessionInfo[], repositories: readonly PiRepository[], mounts: readonly string[]): PiWorkspaceGroup[] {
  const groups = new Map<string, PiWorkspaceGroup>()
  for (const session of sessions) {
    const repository = piSessionRepository(session, repositories)
    if (!repository || mounts.some((mount) => isInside(repository.path, mount))) continue
    const group = groups.get(repository.path) ?? { path: trimSlash(repository.path), label: repository.name, sessions: [] }
    group.sessions.push(session)
    groups.set(repository.path, group)
  }
  return [...groups.values()].sort((left, right) => PATH_COLLATOR.compare(left.label, right.label))
}

/** A session's dot: its tone, whether only its ring shows, and its accessible name. */
export interface PiStatusMark {
  tone: StatusTone
  hollow: boolean
  label: string
}

type MarkedStatus = Pick<PiSessionStatus, "running" | "waiting" | "unread"> & Partial<Pick<PiSessionStatus, "failed" | "cachedUntil">>

/**
 * The dot of a session's state at `now`, null while idle, read and cold.
 *
 * - Waiting, running (pulsing) and failed win
 * - Otherwise filled means unread, hollow read; green means ready: the prompt cache is warm, so going on is cheap
 */
export function piStatusMark(status: MarkedStatus, now = Date.now()): PiStatusMark | null {
  if (status.waiting) return { tone: "attention", hollow: false, label: "Waiting" }
  if (status.running) return { tone: "active", hollow: false, label: "Running" }
  if (status.failed) return { tone: "error", hollow: false, label: "Failed" }
  const ready = status.cachedUntil !== undefined && status.cachedUntil > now
  if (status.unread) return ready ? { tone: "done", hollow: false, label: "Unread, ready" } : { tone: "ready", hollow: false, label: "Unread" }
  return ready ? { tone: "done", hollow: true, label: "Ready" } : null
}

/** Rank of a mark for the one a group of sessions shows: what needs the user first. */
const MARK_RANK = ["Waiting", "Running", "Failed", "Unread, ready", "Unread", "Ready"]

/** The most pressing mark of `statuses` at `now`, null when none has one. */
export function piGroupMark(statuses: readonly MarkedStatus[], now = Date.now()): PiStatusMark | null {
  const marks = statuses.map((status) => piStatusMark(status, now)).filter((mark) => mark !== null)
  return marks.sort((left, right) => MARK_RANK.indexOf(left.label) - MARK_RANK.indexOf(right.label))[0] ?? null
}

/** The current time, moved on when the earliest of `untils` still ahead passes, so a warm cache turns cold on screen. */
export function useNowUntil(untils: readonly (number | undefined)[]): number {
  const [now, setNow] = useState(Date.now)
  const next = Math.min(...untils.filter((until): until is number => until !== undefined && until > now))
  useEffect(() => {
    if (!Number.isFinite(next)) return
    const timer = window.setTimeout(() => setNow(Date.now()), Math.min(next - Date.now() + 50, 2 ** 31 - 1))
    return () => window.clearTimeout(timer)
  }, [next])
  return now
}

function MarkDot({ mark }: { mark: PiStatusMark | null }) {
  return mark ? <span role="img" aria-label={mark.label} className="flex items-center"><StatusDot tone={mark.tone} hollow={mark.hollow} /></span> : null
}

/** `30% · $0.45` once Pi reported them for an attached session. */
export function piUsageText(status: { contextPercent?: number; cost?: number }): string {
  return [
    status.contextPercent === undefined ? "" : `${Math.round(status.contextPercent)}%`,
    status.cost === undefined ? "" : `$${status.cost.toFixed(2)}`,
  ].filter(Boolean).join(" · ")
}

/** Context size after a session's last answer as `84k`, `<1k` below a thousand, nothing before the first answer. */
export function piTokenText(tokens: number): string {
  if (tokens <= 0) return ""
  return tokens < 1000 ? "<1k" : `${Math.round(tokens / 1000)}k`
}

/** The live dot of a session (`piStatusMark`). */
export function PiSessionDot({ sessionId }: { sessionId: string }) {
  const status = usePiSessionStatus(sessionId)
  return <MarkDot mark={piStatusMark(status, useNowUntil([status.cachedUntil]))} />
}

/** The most pressing dot of `sessionIds` (`piGroupMark`), e.g. on a Sessions section. */
export function PiSessionsDot({ sessionIds }: { sessionIds: readonly string[] }) {
  const summary = usePiSessionsSummary(sessionIds, (statuses) => JSON.stringify(statuses.map((status) => [status.running, status.waiting, status.unread, status.failed, status.cachedUntil])))
  const statuses = useMemo(() => (JSON.parse(summary) as [boolean, boolean, boolean, boolean, number | null][])
    .map(([running, waiting, unread, failed, cachedUntil]) => ({ running, waiting, unread, failed, cachedUntil: cachedUntil ?? undefined })), [summary])
  return <MarkDot mark={piGroupMark(statuses, useNowUntil(statuses.map((status) => status.cachedUntil)))} />
}

/** Row end of a session: worktree branch, pin, context and cost, state dot. */
function SessionMeta({ session, worktree, pinned }: { session: PiSessionInfo; worktree: string | null; pinned: boolean }) {
  const usage = piUsageText(usePiSessionStatus(session.sessionId))
  return <span className="flex items-center gap-1.5 text-ink-faint">
    {worktree && <span className="max-w-20 truncate">{worktree}</span>}
    {pinned && <Pin role="img" aria-label="Pinned" className="h-3 w-3" />}
    {usage && <span className="tabular-nums">{usage}</span>}
    <PiSessionDot sessionId={session.sessionId} />
  </span>
}

/**
 * Sidebar rows for Pi coding sessions, from the app's session store. Workbench `document` chats stay with their file (G5).
 *
 * - Each mount and each checkout outside the mounts gets a Sessions folder, worktree sessions sit under their main checkout
 * - Forks nest under their parent, a row shows its worktree, pin, context use, cost and running/waiting/unread dot
 * - A row opens its session, or switches to the tab already showing it; `+` starts a session in a new tab
 * - Sessions pin from their row's menu and delete with its trash button
 */
export function usePiSidebar({ onOpen, workspaceActions }: {
  onOpen?: (item: WorkspaceItem, inNewTab?: boolean) => void
  /** Extra icon actions on a directory row, keyed by its directory, `reportError` fills the section's error line. */
  workspaceActions?: (directory: string, label: string, reportError: (message: string) => void) => NonNullable<FileTreeItem["actions"]>
}) {
  const { tab } = useTab()
  const listed = usePiSessions({ profile: "coding" }).sessions
  const repositories = usePiRepositories()
  /** Rows deleted before the list reads again. */
  const [deleted, setDeleted] = useState<ReadonlySet<string>>(() => new Set())
  const [pins, setPins] = useState<ReadonlyMap<string, boolean>>(() => new Map())
  const [error, setError] = useState("")
  const shownSessionId = itemOf(tab, "pi")?.pi?.sessionId ?? null
  const sessions = listed.filter((session) => !deleted.has(session.sessionId))

  /** Hide the row at once, bring it back when the delete fails. */
  const remove = async (session: PiSessionInfo) => {
    setError("")
    setDeleted((current) => new Set(current).add(session.sessionId))
    try {
      await deletePiSession(session.sessionId)
      refreshPiSessions()
    } catch (cause) {
      setDeleted((current) => new Set([...current].filter((id) => id !== session.sessionId)))
      setError(piSessionError(cause, "Could not delete this session"))
    }
  }

  const pin = async (session: PiSessionInfo, pinned: boolean) => {
    setError("")
    setPins((current) => new Map(current).set(session.sessionId, pinned))
    try {
      await pinPiSession(session.sessionId, pinned)
      refreshPiSessions()
    } catch (cause) {
      setPins((current) => new Map([...current].filter(([id]) => id !== session.sessionId)))
      setError(piSessionError(cause, "Could not pin this session"))
    }
  }

  const sessionItem = (node: PiSessionNode, repo: string, projectSlug: string | null): FileTreeItem<string> => {
    const { session } = node
    const title = piSessionTitle(session)
    const pinned = pins.get(session.sessionId) ?? session.pinned
    const repository = piSessionRepository(session, repositories)
    return {
      id: `pi:${session.sessionId}`,
      label: title,
      type: node.forks.length > 0 ? "folder" : "element",
      isActive: session.sessionId === shownSessionId,
      meta: <SessionMeta session={session} worktree={piSessionWorktree(session, repository)} pinned={pinned} />,
      onSelect: () => onOpen?.(piSessionItem(session, repository?.name ?? repo, projectSlug)),
      actions: [
        { label: `${pinned ? "Unpin" : "Pin"} ${title}`, menuLabel: pinned ? "Unpin" : "Pin", icon: pinned ? PinOff : Pin, onClick: () => { void pin(session, !pinned) } },
      ],
      onDelete: () => { void remove(session) },
      children: node.forks.map((fork) => sessionItem(fork, repo, projectSlug)),
    }
  }

  /** The "Sessions" folder of a directory row, from `directorySessions` or every session belonging to `directory`. */
  const directoryCode = (directory: string, name: string, projectSlug: string | null, directorySessions: readonly PiSessionInfo[] = piSessionsInside(sessions, directory, repositories)): FileTreeItem<string> => {
    const repo = newPiItem(directory, projectSlug).pi!.repo
    return {
      id: `${directory}/:code`,
      label: "Sessions",
      icon: Bot,
      signal: <PiSessionsDot sessionIds={directorySessions.map((session) => session.sessionId)} />,
      type: "folder",
      create: { label: `New session in ${name}`, onClick: () => onOpen?.(newPiItem(directory, projectSlug), true) },
      childLimit: SESSION_LIMIT,
      children: piSessionForest(directorySessions).map((node) => sessionItem(node, repo, projectSlug)),
    }
  }

  /** Extra icon actions of a directory row. */
  const directoryActions = (directory: string, name: string) => workspaceActions?.(directory, name, setError) ?? []

  /** Every checkout outside all `mounts` holding sessions, with them. */
  const unmountedWorkspaces = (mounts: readonly string[]) => unmountedPiWorkspaces(sessions, repositories, mounts)

  return { error, directoryCode, directoryActions, unmountedWorkspaces }
}
