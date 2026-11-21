"use client"

import { useEffect, useState } from "react"
import { createPortal } from "react-dom"
import { ErrorText } from "@/components/ui/ErrorText"
import { PiSessionView, type PiShellCommands } from "@/features/pi/PiTab"
import { PiPanelBar, type PiCatalogProps, type PiSessionHeader } from "@/features/pi/PiTopBar"
import { openFactoryTaskSession } from "@/lib/backend/api/factory"
import type { FactoryTask } from "@/lib/domain/factory"
import { errorMessage } from "@/lib/utils"

/**
 * The task's own Pi session in the agent panel: its transcript and a composer to steer the coder.
 *
 * - The session is the task's (`session_id`), in the task's checkout. A prompt goes the way of any Pi session
 * - A session whose Pi child is stopped (`live` null) is opened first (`POST /api/factory/tasks/{id}/session`), once:
 *   pi-host attaches only to a live one. Once shown the view stays, it reports a child that exits itself
 * - The session bar renders into `bar`, the panel header
 */
export function TaskPiChat({ task, bar, modelProps, shellCommands }: {
  task: FactoryTask
  bar: HTMLElement | null
  modelProps: PiCatalogProps
  shellCommands?: PiShellCommands
}) {
  const [live, setLive] = useState(task.live !== null)
  const [error, setError] = useState("")
  // The child may come up without this panel: a command of the task view opens the session too
  if (!live && task.live !== null) setLive(true)

  useEffect(() => {
    if (live) return
    let cancelled = false
    openFactoryTaskSession(task.id).then(
      () => cancelled || setLive(true),
      (cause: unknown) => cancelled || setError(errorMessage(cause, "Could not open the agent's session")),
    )
    return () => {
      cancelled = true
    }
  }, [live, task.id])

  const inBar = (header: PiSessionHeader | null) => bar && createPortal(<PiPanelBar header={header} modelProps={modelProps} />, bar)
  const cwd = task.worktree ?? task.repo

  if (!live) {
    return <>
      {inBar(null)}
      <div role="status" className="flex flex-1 items-center justify-center px-4 text-center text-xs text-ink-faint">{error ? <ErrorText>{error}</ErrorText> : "Loading…"}</div>
    </>
  }
  return <PiSessionView sessionId={task.session_id} cwd={cwd} files={{ cwd }} renderHeader={inBar} shellCommands={shellCommands} />
}
