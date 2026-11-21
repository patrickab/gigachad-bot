import type { WorkspaceItem } from "@/contexts/TabContext"

/** A task item's path: the task id, "@", the repo path. An absolute repo path starts with "/", so it never reads as one. */
const TASK_ITEM_PATH = /^([a-z0-9]{8,32})@(\/.*)$/

/** Workspace item that shows the factory board of the repo at path `repo`, or one of its tasks when `taskId` is given. */
export function factoryItem(repo: string, taskId?: string): WorkspaceItem {
  return { kind: "factory", path: taskId ? `${taskId}@${repo}` : repo, projectSlug: null }
}

/** Repo path and task a factory item's path names. */
export function parseFactoryPath(path: string): { repo: string; taskId: string | null } {
  const task = TASK_ITEM_PATH.exec(path)
  return task ? { repo: task[2], taskId: task[1] } : { repo: path, taskId: null }
}
