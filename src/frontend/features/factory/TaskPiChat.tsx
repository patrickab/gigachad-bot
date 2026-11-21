"use client"

import { useEffect, useState } from "react"
import { createPortal } from "react-dom"
import { ErrorText } from "@/components/ui/ErrorText"
import { PiSessionView, type PiShellCommands } from "@/features/pi/PiTab"
import { PiPanelBar, type PiCatalogProps, type PiSessionHeader } from "@/features/pi/PiTopBar"
import { refreshPiSessions } from "@/features/pi/sessionStore"
import { ApiError, openFactoryTaskSession } from "@/lib/backend/api"
import { errorMessage } from "@/lib/utils"
import type { FactoryTaskDetail } from "@/lib/domain/factory"

/** Saved agent sessions only change when a phase run ends, so the session opens again then. */
function savedKey(task: FactoryTaskDetail): number {
  return task.phases.filter((run) => run.status !== "running").length
}

/**
 * The task agent's own Pi session, the agent panel's Pi mode: its transcript and a composer to continue it in the worktree.
 *
 * - The backend copies the newest saved phase session into the worktree's Pi sessions and opens it (`POST /api/factory/tasks/{id}/session`)
 * - The engine saves a session when its agent turn ends: while a phase runs the panel shows the last saved one, and opens the new one when the run ends
 * - The session bar renders into `bar`, the panel header
 */
export function TaskPiChat({ task, bar, modelProps, shellCommands }: {
  task: FactoryTaskDetail
  bar: HTMLElement | null
  modelProps: PiCatalogProps
  shellCommands?: PiShellCommands
}) {
  const [opened, setOpened] = useState<{ sessionId: string; cwd: string; key: number } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const key = savedKey(task)
  const running = task.phases.find((run) => run.status === "running")

  useEffect(() => {
    let cancelled = false
    openFactoryTaskSession(task.id).then(
      ({ session }) => {
        if (cancelled) return
        setOpened({ sessionId: session.sessionId, cwd: session.cwd, key })
        setError(null)
        refreshPiSessions()
      },
      (cause: unknown) => {
        if (cancelled) return
        setError(cause instanceof ApiError && cause.status === 404 ? "No agent session yet" : errorMessage(cause, "Could not open the agent's session"))
      },
    )
    return () => { cancelled = true }
  }, [task.id, key])

  const inBar = (header: PiSessionHeader | null) => bar && createPortal(<PiPanelBar header={header} modelProps={modelProps} />, bar)

  if (!opened) {
    return <>
      {inBar(null)}
      <div role="status" className="flex flex-1 items-center justify-center text-xs text-ink-faint">{error ? (error === "No agent session yet" ? error : <ErrorText>{error}</ErrorText>) : "Loading…"}</div>
    </>
  }
  return <>
    {running && <p className="shrink-0 px-4 pt-2 text-[11px] text-ink-subtle">Agent is running, showing its last saved turn</p>}
    <PiSessionView
      key={`${opened.sessionId}:${opened.key}`}
      sessionId={opened.sessionId}
      cwd={opened.cwd}
      files={{ cwd: opened.cwd }}
      renderHeader={inBar}
      shellCommands={shellCommands}
    />
  </>
}
