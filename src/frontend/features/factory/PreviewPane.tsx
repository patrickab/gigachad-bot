"use client"

import { ExternalLink, RotateCw } from "lucide-react"
import { ErrorText } from "@/components/ui/ErrorText"
import { useAsyncAction } from "@/hooks/useAsyncAction"
import { restartFactoryPreview } from "@/lib/backend/api"
import { allowedActions } from "./states"
import { ICON_BUTTON_CLASS } from "./styles"
import type { FactoryTaskDetail } from "@/lib/domain/types"

function Message({ children }: { children: React.ReactNode }) {
  return <div className="flex h-full items-center justify-center px-6 text-center text-xs text-ink-faint">{children}</div>
}

/** Live preview of a task's worktree: the engine's preview URL in a frame, with restart and open-in-new-tab. */
export function PreviewPane({ task, onChanged }: { task: FactoryTaskDetail; onChanged: () => void }) {
  const { preview } = task
  const canRestart = allowedActions(task.state).restartPreview
  const { run: restart, pending: restarting, error } = useAsyncAction(async () => {
    await restartFactoryPreview(task.id)
    onChanged()
  }, "Could not restart the preview")

  if (!preview) return <Message>No preview for this task</Message>

  return (
    <div className="flex h-full flex-col">
      <div className="flex h-7 shrink-0 items-center gap-1 px-3 text-[10px] text-ink-faint">
        <span className="min-w-0 flex-1 truncate">{preview.status === "ready" ? preview.open_url : preview.status}</span>
        {canRestart && (
          <button type="button" aria-label="Restart preview" disabled={restarting || preview.status === "starting"} onClick={() => void restart()} className={ICON_BUTTON_CLASS}>
            <RotateCw className="h-3.5 w-3.5" />
          </button>
        )}
        {preview.status === "ready" && (
          <a href={preview.open_url} target="_blank" rel="noreferrer" aria-label="Open preview in a new tab" className={ICON_BUTTON_CLASS}>
            <ExternalLink className="h-3.5 w-3.5" />
          </a>
        )}
      </div>
      <ErrorText className="px-3 py-1">{error}</ErrorText>
      <div className="min-h-0 flex-1">
        {preview.status === "ready" ? (
          <iframe
            key={preview.open_url}
            title="Task preview"
            src={preview.open_url}
            sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-downloads"
            className="h-full w-full border-0 bg-paper"
          />
        ) : preview.status === "starting" ? <Message>Starting the preview</Message>
        : preview.status === "failed" ? <Message><span className="text-danger">{preview.error ?? "The preview failed to start"}</span></Message>
        : <Message>The preview is stopped</Message>}
      </div>
    </div>
  )
}
