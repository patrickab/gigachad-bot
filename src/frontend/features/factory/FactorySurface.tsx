"use client"

import { useTab, type WorkspaceItem } from "@/contexts/TabContext"
import { FactoryBoard } from "./FactoryBoard"
import { factoryItem, parseFactoryPath } from "./items"
import { TaskView } from "./TaskView"

/** Public entry of the factory feature: a repo's board, or the task a `factory` workspace item names. */
export function FactorySurface({ item }: { item: WorkspaceItem }) {
  const { setWorkspaceItem } = useTab()
  const { repo, taskId } = parseFactoryPath(item.path)
  return taskId ? (
    <TaskView key={taskId} repo={repo} taskId={taskId} onBack={() => setWorkspaceItem(factoryItem(repo))} />
  ) : (
    <FactoryBoard key={repo} repo={repo} onOpenTask={(id) => setWorkspaceItem(factoryItem(repo, id))} />
  )
}
