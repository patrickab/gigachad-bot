import type { OmpOverview, OmpSession, OmpView } from "@/lib/domain/types"
import { PATH_COLLATOR, baseName } from "@/lib/paths"

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

function trimSlash(path: string): string {
  return path.length > 1 ? path.replace(/\/+$/, "") : path
}

/** Whether `path` is `root` or below it, `/a/foobar` is not inside `/a/foo`. */
export function isInside(path: string, root: string): boolean {
  const [inner, outer] = [trimSlash(path), trimSlash(root)]
  return inner === outer || inner.startsWith(outer === "/" ? "/" : `${outer}/`)
}

/** Directory a session belongs to: its main repository for a worktree cwd, else the cwd. */
function sessionDirectory(session: OmpSession): string {
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
