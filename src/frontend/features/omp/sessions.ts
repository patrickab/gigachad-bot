import type { Pair } from "@/lib/domain/pairs"
import type { Message, OmpOverview, OmpSession, OmpView } from "@/lib/domain/types"
import { PATH_COLLATOR, baseName, isInside, trimSlash } from "@/lib/paths"

/** Sidebar and tab title of a session: its name, else its first prompt. */
export function sessionTitle(session: Pick<OmpSession, "name" | "first_message">): string {
  return session.name || session.first_message.trim() || "Untitled session"
}

/** Tab title for what an omp tab shows. */
export function viewTitle(view: OmpView, overview: OmpOverview | null): string {
  if (view.kind === "settings") return "Settings"
  if (view.kind === "new") return "New session"
  const session = overview?.sessions.find((candidate) => candidate.id === view.sessionId)
  return session ? sessionTitle(session) : "Session"
}

/** Directory a session belongs to: its main repository for a worktree cwd, else the cwd. */
export function sessionDirectory(session: OmpSession): string {
  return session.project_root ?? session.cwd
}

/** File mount linked to a project: the mount whose folder name is the project slug (same rule as the backend's architecture store). */
export function projectMount(slug: string, mounts: readonly string[]): string | null {
  return mounts.find((mount) => baseName(trimSlash(mount)) === slug) ?? null
}

export function sessionsInside(sessions: readonly OmpSession[], directory: string): OmpSession[] {
  return sessions.filter((session) => isInside(sessionDirectory(session), directory))
}

export interface SessionNode {
  session: OmpSession
  forks: SessionNode[]
}

const byNewest = (left: OmpSession, right: OmpSession) => right.modified.localeCompare(left.modified)

/** Nest forks under their parent session; a fork whose parent is not in `sessions` is a root. Newest first. */
export function sessionForest(sessions: readonly OmpSession[]): SessionNode[] {
  const nodes = new Map(sessions.map((session) => [session.id, { session, forks: [] as SessionNode[] }]))
  const roots: SessionNode[] = []
  for (const session of [...sessions].sort(byNewest)) {
    const node = nodes.get(session.id)!
    const parent = session.parent_session_id ? nodes.get(session.parent_session_id) : undefined
    if (parent && parent !== node) parent.forks.push(node)
    else roots.push(node)
  }
  return roots
}

/** How long a session that stopped running stays in the picker's "Recent" group. */
export const RECENT_SESSION_MS = 24 * 60 * 60 * 1000

export interface SessionGroups {
  running: OmpSession[]
  recent: OmpSession[]
  older: OmpSession[]
}

/** `sessions` split into running ones, the rest modified within `RECENT_SESSION_MS` of `nowMs`, and the older rest. Each newest first. */
export function groupSessions(sessions: readonly OmpSession[], runningIds: readonly string[], nowMs: number): SessionGroups {
  const running = new Set(runningIds)
  const groups: SessionGroups = { running: [], recent: [], older: [] }
  for (const session of [...sessions].sort(byNewest)) {
    if (running.has(session.id)) groups.running.push(session)
    else if (nowMs - Date.parse(session.modified) <= RECENT_SESSION_MS) groups.recent.push(session)
    else groups.older.push(session)
  }
  return groups
}

/** A session history (`fetchOmpSessionHistory`) as chat pairs: each prompt with every assistant text up to the next prompt, omp splits one answer across tool steps. */
export function historyPairs(history: readonly Pick<Message, "role" | "content">[]): Pair[] {
  const pairs: Pair[] = []
  history.forEach((message, index) => {
    const last = pairs.at(-1)
    if (message.role !== "user" && last) {
      last.assistant = { role: "assistant", content: [last.assistant.content, message.content].filter(Boolean).join("\n\n") }
      return
    }
    const isPrompt = message.role === "user"
    pairs.push({
      user: { role: "user", content: isPrompt ? message.content : "" },
      assistant: { role: "assistant", content: isPrompt ? "" : message.content },
      globalIndex: index,
      qaIndex: pairs.length,
    })
  })
  return pairs
}

interface WorkspaceGroup {
  path: string
  label: string
  sessions: OmpSession[]
}

/**
 * Group sessions outside every `linkedPaths` entry by omp-web workspace.
 *
 * - A session joins the deepest workspace containing it, else a group of its own directory
 * - Workspaces inside a linked path are left out, a project's code category lists them
 */
export function unlinkedWorkspaces(overview: OmpOverview, linkedPaths: readonly string[]): WorkspaceGroup[] {
  const isLinked = (path: string) => linkedPaths.some((linked) => isInside(path, linked))
  const aliases = new Map(overview.workspaces.map((workspace) => [trimSlash(workspace.path), workspace.alias]))
  const groups = new Map<string, OmpSession[]>()
  for (const path of aliases.keys()) if (!isLinked(path)) groups.set(path, [])
  for (const session of overview.sessions) {
    const directory = trimSlash(sessionDirectory(session))
    if (isLinked(directory)) continue
    const workspace = [...groups.keys()].filter((path) => aliases.has(path) && isInside(directory, path))
      .sort((left, right) => right.length - left.length)[0] ?? directory
    groups.set(workspace, [...(groups.get(workspace) ?? []), session])
  }
  return [...groups.entries()]
    .map(([path, sessions]) => ({ path, label: aliases.get(path) || baseName(path), sessions }))
    .sort((left, right) => PATH_COLLATOR.compare(left.label, right.label) || PATH_COLLATOR.compare(left.path, right.path))
}

/**
 * Directories an omp session can start in: every mount and omp-web workspace, the most recently used first, the rest by name.
 *
 * - A directory counts as used when a session ran in it or below it (a worktree counts for its main repository)
 */
export function startDirectories(mounts: readonly string[], overview: OmpOverview | null): string[] {
  const workspaces = overview ? unlinkedWorkspaces(overview, mounts).map((workspace) => workspace.path) : []
  const directories = [...new Set([...mounts.map(trimSlash), ...workspaces])]
  const sessions = overview?.sessions ?? []
  const used = new Map(directories.map((directory) => [directory, sessionsInside(sessions, directory).map((session) => session.modified).sort().at(-1) ?? ""]))
  return directories.sort((left, right) => used.get(right)!.localeCompare(used.get(left)!) || PATH_COLLATOR.compare(baseName(left), baseName(right)))
}
