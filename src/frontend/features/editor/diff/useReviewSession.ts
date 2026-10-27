import { useMemo, useState } from "react"
import type { RootDiffFiles } from "@/lib/domain/files"
import { errorMessage } from "@/lib/utils"
import { formatReview, useReviewNotes, type NoteAnchor, type ReviewNote } from "./review"
import type { FilePatch } from "./useRootDiff"

/** Review delivery callback and receiver availability. */
export type ReviewTarget = { send: (text: string) => Promise<void>; canSend: boolean }

/** New note when noteId is null, otherwise an existing note. */
export type Draft = { anchor: NoteAnchor; noteId: string | null; comment: string }

export type ReviewActions = {
  startDraft: (draft: Draft) => void
  saveDraft: (draft: Draft, comment: string) => void
  cancelDraft: () => void
  removeNote: (id: string) => void
}

export const NO_NOTES: ReviewNote[] = []

/** Persisted notes of `root`, the open draft, and delivery to `review` or the clipboard. */
export function useReviewSession(root: string, review: ReviewTarget | undefined, files: RootDiffFiles["files"], patches: Record<string, FilePatch>) {
  const { notes, update: updateNotes } = useReviewNotes(root)
  const [draft, setDraft] = useState<Draft | null>(null)
  const [sending, setSending] = useState(false)
  const [sendError, setSendError] = useState("")
  const [copied, setCopied] = useState(false)

  const actions = useMemo<ReviewActions>(() => ({
    startDraft: setDraft,
    cancelDraft: () => setDraft(null),
    saveDraft: (saved, comment) => {
      const text = comment.trim()
      updateNotes((current) => saved.noteId === null
        ? text ? [...current, { ...saved.anchor, id: crypto.randomUUID(), comment: text }] : current
        : text ? current.map((note) => note.id === saved.noteId ? { ...note, comment: text } : note) : current.filter((note) => note.id !== saved.noteId))
      setDraft(null)
    },
    removeNote: (id) => updateNotes((current) => current.filter((note) => note.id !== id)),
  }), [updateNotes])
  const notesByPath = useMemo(() => {
    const grouped = new Map<string, ReviewNote[]>()
    for (const note of notes) grouped.set(note.path, [...(grouped.get(note.path) ?? []), note])
    return grouped
  }, [notes])

  const send = async () => {
    if ((review && !review.canSend) || sending || notes.length === 0) return
    const sent = new Set(notes)
    setSending(true)
    setSendError("")
    setCopied(false)
    try {
      const currentPatches = Object.fromEntries(files.map((file) => {
        const patch = patches[file.path]
        return [file.path, patch?.stamp === file.stamp ? patch.lines : null]
      }))
      const text = formatReview(notes, currentPatches)
      if (review) {
        await review.send(text)
        updateNotes((current) => current.filter((note) => !sent.has(note)))
      } else {
        await navigator.clipboard.writeText(text)
        setCopied(true)
      }
    } catch (error) {
      setSendError(errorMessage(error, "Could not export the review"))
    } finally {
      setSending(false)
    }
  }

  return { notes, notesByPath, draft, setDraft, actions, send, sending, sendError, copied }
}
