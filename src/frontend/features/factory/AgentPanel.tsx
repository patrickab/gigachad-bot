"use client"

import { useState } from "react"
import { Composer, COMPOSER_ICON_BUTTON } from "@/components/ui/Composer"
import { ErrorText } from "@/components/ui/ErrorText"
import { FilePanel } from "@/features/editor/FilePanel"
import { useAsyncAction } from "@/hooks/useAsyncAction"
import { callFactory } from "@/lib/backend/api"
import type { FactoryTask, FactoryTimelineEntry } from "@/lib/domain/factory"
import { AttachImageButton, ImageChips, useRequestImages } from "./requestImages"
import { TaskTimeline } from "./TaskTimeline"

const PANEL_WIDTH_PX = 420

/**
 * A task's agent in the tab's file panel, beside the diff: its timeline and a message box to the coder, none for a
 * merge request (no coder runs it).
 *
 * - The timeline comes from `entries` (the owner reads them, the diff follows them too)
 * - A message goes to the core (`tasks.message`): a resting task runs new rounds, a running coder gets it as a
 *   follow-up. Images (picked or pasted) are stored on the host and their paths join the text. Enter sends, Shift+Enter adds a line; disabled while `canMessage` is false (the core refuses `tasks.message`). In
 *   `waiting_answer` the message answers the coder's question
 * - `onSent` runs after the core took a message, so the owner reads the task again
 * - `onOpenTask` opens a task the coder started
 */
export function AgentPanel({ task, entries, timelineError, canMessage, onSent, onOpenTask, onClose }: {
  task: FactoryTask
  entries: readonly FactoryTimelineEntry[]
  timelineError: string
  canMessage: boolean
  onSent: () => void
  onOpenTask: (taskId: string) => void
  onClose: () => void
}) {
  const [text, setText] = useState("")
  const pictures = useRequestImages()
  const { run: send, pending, error } = useAsyncAction(async (message: string) => {
    await callFactory("tasks.message", { id: task.id, text: await pictures.attach(message) })
    setText("")
    pictures.clear()
    onSent()
  }, "Could not send the message")

  return (
    <FilePanel label="Agent" bare onClose={onClose} width={PANEL_WIDTH_PX}>
      <div className="flex min-h-0 flex-1 flex-col">
        <TaskTimeline entries={entries} usage={task.usage} error={timelineError} onOpenTask={onOpenTask} />
        {task.kind !== "merge_request" && <div className="shrink-0 px-3 pb-3">
          <ErrorText className="px-1 pb-1 text-[11px]">{error}</ErrorText>
          <Composer
            label="Message to the coder"
            value={text}
            onValueChange={setText}
            onSubmit={() => void send(text.trim())}
            disabled={!canMessage}
            busy={pending}
            canSend={text.trim().length > 0 || pictures.images.length > 0}
            onPaste={pictures.onPaste}
            leading={<AttachImageButton onPick={pictures.add} disabled={!canMessage} className={COMPOSER_ICON_BUTTON} />}
          >
            <ImageChips images={pictures.images} onRemove={pictures.remove} />
          </Composer>
        </div>}
      </div>
    </FilePanel>
  )
}
