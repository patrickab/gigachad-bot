// Review comments on a task's diff. Consumes the editor diff's review session (notes, drafts, delivery) and
// `sendFactoryTaskMessage`, provides `formatTaskReview` (the message a review sends) and `useTaskReview` (the pending
// comments of one task, sent as one message that opens a round).
import { useEffect, useMemo, useRef } from "react"
import type { ReviewNote } from "@/features/editor/diff/review"
import { useReviewSession, type ReviewTarget } from "@/features/editor/diff/useReviewSession"
import type { FilePatch } from "@/features/editor/diff/useRootDiff"
import { sendFactoryTaskMessage } from "@/lib/backend/api"
import { allowedActions, FINISHED_STATES, type FactoryShownState } from "./states"
import type { TaskDiffFile } from "./useTaskDiff"

/** A fence longer than any backtick run in `text`, so the quoted line cannot close it. */
function fenceFor(text: string): string {
  const longest = Math.max(0, ...(text.match(/`+/g) ?? []).map((run) => run.length))
  return "`".repeat(Math.max(3, longest + 1))
}

/**
 * The message of a review: a first line, then one block per comment in path and line order: `path:line`, the quoted
 * line as a fenced snippet, the comment. A comment keeps the line it was written on, also when the diff moved since.
 */
export function formatTaskReview(notes: readonly ReviewNote[]): string {
  const blocks = [...notes]
    .sort((a, b) => a.path.localeCompare(b.path) || a.line - b.line)
    .map((note) => {
      const fence = fenceFor(note.lineText)
      return [`${note.path}:${note.line}`, fence, note.lineText, fence, note.comment].join("\n")
    })
  return ["Review of the diff:", ...blocks].join("\n\n")
}

const NO_FILES: TaskDiffFile[] = []

/** Storage key of a task's pending comments, they stay while the user moves between board and task. */
const reviewKey = (taskId: string) => `factory-task:${taskId}`

/**
 * Pending review comments of task `taskId` on its diff `files`, shown as `state`.
 *
 * - Sendable unless the task merges (the backend refuses a message then) or is still starting
 * - A sent review clears the comments it sent, a failed one keeps them and reports the error
 * - Comments are discarded when the task finishes while it is shown, as its diff then no longer moves
 */
export function useTaskReview(taskId: string, state: FactoryShownState | undefined, files: TaskDiffFile[] | undefined) {
  const shownFiles = files ?? NO_FILES
  const patches = useMemo<Record<string, FilePatch>>(() => Object.fromEntries(shownFiles.map((file) => [file.path, file.patch])), [shownFiles])
  const target: ReviewTarget = {
    canSend: state !== undefined && state !== "merging" && allowedActions(state).message,
    format: formatTaskReview,
    send: async (text) => {
      await sendFactoryTaskMessage(taskId, text)
    },
  }
  const session = useReviewSession(reviewKey(taskId), target, shownFiles, patches)

  // Known and unfinished before, finished now: a task first shown finished keeps the comments written on it
  const finished = state === undefined ? undefined : FINISHED_STATES.includes(state)
  const wasFinished = useRef(finished)
  const { notes, actions } = session
  useEffect(() => {
    if (finished && wasFinished.current === false) for (const note of notes) actions.removeNote(note.id)
    if (finished !== undefined) wasFinished.current = finished
  }, [finished, notes, actions])

  return { ...session, patches, canSend: target.canSend }
}
