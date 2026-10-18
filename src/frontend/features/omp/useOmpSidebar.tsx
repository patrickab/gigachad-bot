import { useCallback, useEffect, useState } from "react"
import { Archive, RotateCcw } from "lucide-react"
import { useTab } from "@/contexts/TabContext"
import type { WorkspaceItem } from "@/contexts/TabContext"
import type { FileTreeItem } from "@/components/ui/file-tree/FileTree"
import { ApiError, addOmpWorkspace, archiveOmpSession, fetchOmpArchive, restoreOmpSession } from "@/lib/backend/api"
import type { OmpArchivedSession, OmpSession, OmpView, ProjectListItem } from "@/lib/domain/types"
import { OmpBranchTree } from "./OmpBranchTree"
import { useOmp } from "./OmpProvider"
import { sessionForest, sessionTitle, sessionsInside, unlinkedWorkspaces, type SessionNode } from "./sessions"

/** Workspace item showing `view`, `path` is the workspace directory it was opened from. */
export function ompItem(view: OmpView, path: string, projectSlug: string | null): WorkspaceItem {
  return { kind: "omp", path, projectSlug, omp: view }
}

/** Session rows a code category or omp workspace shows before "Show more", like omp-web's project list. */
const SESSION_LIMIT = 5

/**
 * Sidebar rows for omp-web sessions, built from the shared overview.
 *
 * - Sessions nest their forks, the running ones carry a "running" badge
 * - Selecting a session shows it in the current tab, `+` starts a session in a new tab like a new chat
 * - Sessions archive from their row, the OMP section's "archived" folder restores them
 * - Everything is empty while omp-web never answered
 */
export function useOmpSidebar({ onOpen }: { onOpen?: (item: WorkspaceItem, inNewTab?: boolean) => void }) {
  const { tab, isActive } = useTab()
  const { overview, refreshOverview } = useOmp()
  const [archived, setArchived] = useState<OmpArchivedSession[]>([])
  const [archiving, setArchiving] = useState<ReadonlySet<string>>(() => new Set())
  const [error, setError] = useState("")
  const shownView = tab.appMode === "omp" ? tab.workspaceItem?.omp : undefined
  const shownSessionId = shownView?.kind === "session" ? shownView.sessionId : null
  const running = new Set(overview?.running_session_ids ?? [])
  const sessions = (overview?.sessions ?? []).filter((session) => !archiving.has(session.id))

  const refreshArchive = useCallback(() => {
    fetchOmpArchive().then(setArchived).catch(() => {})
  }, [])

  const available = overview !== null
  useEffect(() => {
    if (isActive && available) refreshArchive()
  }, [isActive, available, refreshArchive])

  const forget = (sessionId: string) => setArchiving((current) => new Set([...current].filter((id) => id !== sessionId)))

  /** Hides the row at once and keeps it hidden, the next overview no longer lists it. */
  const archive = async (session: OmpSession) => {
    setError("")
    setArchiving((current) => new Set(current).add(session.id))
    try {
      await archiveOmpSession(session.id)
      refreshOverview()
      refreshArchive()
    } catch (cause) {
      forget(session.id)
      const forksFirst = cause instanceof ApiError && cause.status === 409
      setError(forksFirst ? `Archive the forks of ${sessionTitle(session)} first` : cause instanceof Error ? cause.message : "Could not archive this session")
    }
  }

  const restore = async (entry: OmpArchivedSession) => {
    setError("")
    try {
      forget(await restoreOmpSession(entry.key))
      refreshOverview()
      refreshArchive()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not restore this session")
    }
  }

  const sessionItem = (node: SessionNode, path: string, projectSlug: string | null): FileTreeItem<string> => {
    const { session } = node
    const title = sessionTitle(session)
    return {
      id: `omp:${session.id}`,
      label: title,
      type: node.forks.length > 0 ? "folder" : "element",
      badge: running.has(session.id) ? "running" : undefined,
      isActive: session.id === shownSessionId,
      detail: session.id === shownSessionId ? <OmpBranchTree sessionId={session.id} /> : undefined,
      onSelect: () => onOpen?.(ompItem({ kind: "session", sessionId: session.id }, path, projectSlug)),
      actions: [{ label: `Archive ${title}`, icon: Archive, onClick: () => { void archive(session) } }],
      children: node.forks.map((fork) => sessionItem(fork, path, projectSlug)),
    }
  }

  /** A project's "code" category: sessions in its linked mount (worktrees included), `+` starts one there. */
  const codeCategory = (project: ProjectListItem, mount: string): FileTreeItem<string> | null => {
    if (!overview) return null
    return {
      id: `project:${project.slug}:code`,
      label: "code",
      type: "folder",
      create: { label: `New session in ${project.name}`, onClick: () => onOpen?.(ompItem({ kind: "new", cwd: mount }, mount, project.slug), true) },
      childLimit: SESSION_LIMIT,
      children: sessionForest(sessionsInside(sessions, mount)).map((node) => sessionItem(node, mount, project.slug)),
    }
  }

  /** OMP section rows: every omp-web workspace no project's code category covers, with its sessions, then the archive. */
  const workspaceItems = (linkedMounts: readonly string[]): FileTreeItem<string>[] => {
    if (!overview) return []
    const workspaces = unlinkedWorkspaces({ ...overview, sessions }, linkedMounts).map((workspace): FileTreeItem<string> => ({
      id: `omp-workspace:${workspace.path}`,
      label: workspace.label,
      type: "root",
      create: { label: `New session in ${workspace.label}`, onClick: () => onOpen?.(ompItem({ kind: "new", cwd: workspace.path }, workspace.path, null), true) },
      childLimit: SESSION_LIMIT,
      children: sessionForest(workspace.sessions).map((node) => sessionItem(node, workspace.path, null)),
    }))
    if (archived.length === 0) return workspaces
    return [...workspaces, {
      id: "omp-archived",
      label: "archived",
      type: "folder",
      children: archived.map((entry) => {
        const title = entry.name || entry.first_message.trim() || "Untitled session"
        return { id: `omp-archived:${entry.key}`, label: title, type: "element", actions: [{ label: `Restore ${title}`, icon: RotateCcw, onClick: () => { void restore(entry) } }] }
      }),
    }]
  }

  const openSettings = () => onOpen?.(ompItem({ kind: "settings" }, "", null))

  const addWorkspace = async (path: string) => {
    await addOmpWorkspace(path)
    refreshOverview()
  }

  return { available, error, codeCategory, workspaceItems, openSettings, addWorkspace }
}
