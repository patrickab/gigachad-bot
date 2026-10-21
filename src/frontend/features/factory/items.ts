import type { WorkspaceItem } from "@/contexts/TabContext"

/** Workspace item that shows a repo's factory board, or one of its tasks when `taskId` is given. */
export function factoryItem(repo: string, taskId?: string): WorkspaceItem {
  return { kind: "factory", path: taskId ? `${repo}/${taskId}` : repo, projectSlug: null }
}

/** Repo and task a factory item's path names; neither repo names nor task ids contain "/". */
export function parseFactoryPath(path: string): { repo: string; taskId: string | null } {
  const [repo, taskId] = path.split("/", 2)
  return { repo, taskId: taskId || null }
}
