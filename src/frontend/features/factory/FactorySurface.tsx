"use client"

import { useTab, type WorkspaceItem } from "@/contexts/TabContext"
import { FactoryBoard } from "./FactoryBoard"
import { factoryItem, parseFactoryPath } from "./items"
import { TaskView } from "./TaskView"
import { useFactoryTasks } from "./useFactoryTasks"

/** Public entry of the factory feature: a repo's board, or the task a `factory` workspace item names. */
export function FactorySurface({ item }: { item: WorkspaceItem }) {
  const { setWorkspaceItem } = useTab()
  const { repo, taskId } = parseFactoryPath(item.path)
  // Read here, so the task view's `useFactoryTask` stays current as well as the board.
  const { tasks, error, refresh } = useFactoryTasks(repo)
  return taskId ? (
    <TaskView key={taskId} repo={repo} taskId={taskId} onBack={() => setWorkspaceItem(factoryItem(repo))} />
  ) : (
    <FactoryBoard key={repo} repo={repo} tasks={tasks} error={error} refresh={refresh} onOpenTask={(id) => setWorkspaceItem(factoryItem(repo, id))} />
  )
}
