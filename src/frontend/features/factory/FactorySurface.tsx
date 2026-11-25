"use client"

import { useTab, type WorkspaceItem } from "@/contexts/TabContext"
import { piSessionItem } from "@/features/pi/usePiSidebar"
import { FactoryBoard } from "./FactoryBoard"
import { factoryItem, parseFactoryPath } from "./items"
import { TaskView } from "./TaskView"
import { useFactoryTasks } from "./useFactoryTasks"

/** Public entry of the factory feature: a repo's board, or the task a `factory` workspace item names. */
export function FactorySurface({ item }: { item: WorkspaceItem }) {
  const { setWorkspaceItem } = useTab()
  const { repo, taskId } = parseFactoryPath(item.path)
  // Read here, so the task view's `useFactoryTask` stays current as well as the board.
  const { tasks, waves, error, refresh } = useFactoryTasks(repo)
  const openTask = (id: string) => setWorkspaceItem(factoryItem(repo, id))
  const openSession = (sessionId: string) => setWorkspaceItem(piSessionItem({ sessionId }, repo, null))
  return taskId ? (
    <TaskView key={taskId} repo={repo} taskId={taskId} onBack={() => setWorkspaceItem(factoryItem(repo))} onOpenTask={openTask} />
  ) : (
    <FactoryBoard key={repo} repo={repo} tasks={tasks} waves={waves} error={error} refresh={refresh} onOpenTask={openTask} onOpenSession={openSession} />
  )
}
