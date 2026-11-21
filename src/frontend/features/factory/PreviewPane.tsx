"use client"

import { useEffect, useState } from "react"
import { ExternalLink, Square } from "lucide-react"
import { ErrorText } from "@/components/ui/ErrorText"
import { useAsyncAction } from "@/hooks/useAsyncAction"
import { getFactoryPreview, startFactoryPreview, stopFactoryPreview } from "@/lib/backend/api"
import type { FactoryPreview } from "@/lib/domain/factory"
import { errorMessage } from "@/lib/utils"
import { ICON_BUTTON_CLASS, PRIMARY_BUTTON_CLASS } from "./styles"

export const PREVIEW_POLL_MS = 2000

function Centered({ children }: { children: React.ReactNode }) {
  return <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center text-xs text-ink-faint">{children}</div>
}

/**
 * A task's preview, started on demand: the task's checkout served by the backend, shown in a frame.
 *
 * - Reads the preview once (`undefined` until then, `null` when none was started)
 * - None or `stopped`: a start control. `starting`: read again every `PREVIEW_POLL_MS` until it is ready or failed.
 *   `ready`: the frame at `url`, stop and open-in-new-tab. `failed`: the error tail and the start control
 */
export function PreviewPane({ taskId }: { taskId: string }) {
  const [preview, setPreview] = useState<FactoryPreview | null>()
  const [polls, setPolls] = useState(0)
  const [readError, setReadError] = useState("")
  const start = useAsyncAction(async () => setPreview(await startFactoryPreview(taskId)), "Could not start the preview")
  const stop = useAsyncAction(async () => setPreview(await stopFactoryPreview(taskId)), "Could not stop the preview")
  const starting = preview?.state === "starting"

  useEffect(() => {
    let cancelled = false
    getFactoryPreview(taskId).then(
      (answer) => cancelled || setPreview(answer),
      (cause: unknown) => {
        if (cancelled) return
        setPreview(null)
        setReadError(errorMessage(cause, "Could not read the preview"))
      },
    )
    return () => {
      cancelled = true
    }
  }, [taskId])

  // Each answer, or a failed read, counts as a poll and schedules the next one while the preview still starts
  useEffect(() => {
    if (!starting) return
    let cancelled = false
    const timer = setTimeout(() => {
      getFactoryPreview(taskId).then(
        (answer) => {
          if (cancelled) return
          setPreview(answer)
          setReadError("")
        },
        (cause: unknown) => cancelled || setReadError(errorMessage(cause, "Could not read the preview")),
      ).finally(() => cancelled || setPolls((count) => count + 1))
    }, PREVIEW_POLL_MS)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [taskId, starting, polls])

  const error = start.error || stop.error || readError
  const startButton = <button type="button" disabled={start.pending} onClick={() => void start.run()} className={PRIMARY_BUTTON_CLASS}>Start preview</button>

  if (preview === undefined) return <Centered>Loading preview</Centered>
  if (preview === null || preview.state === "stopped") return <Centered>{startButton}<ErrorText>{error}</ErrorText></Centered>
  if (preview.state === "starting") return <Centered><span role="status">Starting the preview</span><ErrorText>{error}</ErrorText></Centered>
  if (preview.state === "failed" || preview.url === null) {
    return (
      <Centered>
        <pre aria-label="Preview error" className="max-h-64 max-w-full overflow-auto whitespace-pre-wrap break-words text-left font-mono text-[11px] text-danger scrollbar-thin">{preview.error ?? "The preview failed to start"}</pre>
        {startButton}
        <ErrorText>{error}</ErrorText>
      </Centered>
    )
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex h-7 shrink-0 items-center gap-1 px-3 text-[10px] text-ink-faint">
        <span className="min-w-0 flex-1 truncate">{preview.url}</span>
        <button type="button" aria-label="Stop preview" disabled={stop.pending} onClick={() => void stop.run()} className={ICON_BUTTON_CLASS}>
          <Square className="h-3.5 w-3.5" />
        </button>
        <a href={preview.url} target="_blank" rel="noreferrer" aria-label="Open preview in a new tab" className={ICON_BUTTON_CLASS}>
          <ExternalLink className="h-3.5 w-3.5" />
        </a>
      </div>
      <ErrorText className="px-3 py-1">{error}</ErrorText>
      <iframe
        key={preview.url}
        title="Task preview"
        src={preview.url}
        sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-downloads"
        className="min-h-0 w-full flex-1 border-0 bg-paper"
      />
    </div>
  )
}
