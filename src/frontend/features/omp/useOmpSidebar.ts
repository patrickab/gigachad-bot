import { useTab } from "@/contexts/TabContext"
import type { WorkspaceItem } from "@/contexts/TabContext"
import type { FileTreeItem } from "@/components/ui/file-tree/FileTree"
import { addOmpWorkspace } from "@/lib/backend/api"
import type { OmpView, ProjectListItem } from "@/lib/domain/types"
import { useOmp } from "./OmpProvider"
import { sessionForest, sessionTitle, sessionsInside, unlinkedWorkspaces, type SessionNode } from "./sessions"

/** Workspace item showing `view`, `path` is the workspace directory it was opened from. */
export function ompItem(view: OmpView, path: string, projectSlug: string | null): WorkspaceItem {
  return { kind: "omp", path, projectSlug, omp: view }
}

/**
 * Sidebar rows for omp-web sessions, built from the shared overview.
 *
 * - Sessions nest their forks, the running ones carry a "running" badge
 * - Selecting a session shows it in the current tab, `+` starts a session in a new tab like a new chat
 * - Everything is empty while omp-web never answered
 */
export function useOmpSidebar({ onOpen }: { onOpen?: (item: WorkspaceItem, inNewTab?: boolean) => void }) {
  const { tab } = useTab()
  const { overview, refreshOverview } = useOmp()
  const shownView = tab.appMode === "omp" ? tab.workspaceItem?.omp : undefined
  const shownSessionId = shownView?.kind === "session" ? shownView.sessionId : null
  const running = new Set(overview?.running_session_ids ?? [])

  const sessionItem = (node: SessionNode, path: string, projectSlug: string | null): FileTreeItem<string> => {
    const { session } = node
    return {
      id: `omp:${session.id}`,
      label: sessionTitle(session),
      type: node.forks.length > 0 ? "folder" : "element",
      badge: running.has(session.id) ? "running" : undefined,
      isActive: session.id === shownSessionId,
      onSelect: () => onOpen?.(ompItem({ kind: "session", sessionId: session.id }, path, projectSlug)),
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
      children: sessionForest(sessionsInside(overview.sessions, mount)).map((node) => sessionItem(node, mount, project.slug)),
    }
  }

  /** OMP section rows: every omp-web workspace no project's code category covers, with its sessions. */
  const workspaceItems = (linkedMounts: readonly string[]): FileTreeItem<string>[] => {
    if (!overview) return []
    return unlinkedWorkspaces(overview, linkedMounts).map((workspace) => ({
      id: `omp-workspace:${workspace.path}`,
      label: workspace.label,
      type: "root",
      create: { label: `New session in ${workspace.label}`, onClick: () => onOpen?.(ompItem({ kind: "new", cwd: workspace.path }, workspace.path, null), true) },
      children: sessionForest(workspace.sessions).map((node) => sessionItem(node, workspace.path, null)),
    }))
  }

  const openSettings = () => onOpen?.(ompItem({ kind: "settings" }, "", null))

  const addWorkspace = async (path: string) => {
    await addOmpWorkspace(path)
    refreshOverview()
  }

  return { available: overview !== null, codeCategory, workspaceItems, openSettings, addWorkspace }
}
