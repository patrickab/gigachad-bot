import { useState } from "react"
import { Pencil, X } from "lucide-react"
import { cn } from "@/lib/utils"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import type { ReviewNote, ReviewSide } from "./review"
import type { Draft, ReviewActions } from "./useReviewSession"

const ICON_BUTTON = "shrink-0 rounded p-1 text-ink-faint hover:bg-hover hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"

/** Distinguish old/new line numbers in accessible labels. */
function lineName(path: string, side: ReviewSide, line: number | null): string {
  return `${path} ${side === "old" ? "old line" : "line"} ${line}`
}

function NoteEditor({ draft, actions }: { draft: Draft; actions: ReviewActions }) {
  const [comment, setComment] = useState(draft.comment)
  const [focused, setFocused] = useState(false)
  // While the textarea has focus Escape discards the draft first, ahead of any enclosing overlay.
  useEscapeLayer(actions.cancelDraft, focused)
  const selectedText = draft.anchor.selectedText
  return <div className="rounded-lg border border-divider bg-surface px-3 py-2 font-sans focus-within:border-ink-muted">
    {selectedText && <pre className="mb-2 max-h-32 overflow-auto whitespace-pre-wrap font-mono text-[11px] text-ink-muted">{selectedText}</pre>}
    <textarea
      autoFocus
      aria-label={`Note on ${lineName(draft.anchor.path, draft.anchor.side, draft.anchor.line)}`}
      rows={2}
      value={comment}
      onChange={(event) => setComment(event.target.value)}
      onFocus={() => setFocused(true)}
      onBlur={() => setFocused(false)}
      onKeyDown={(event) => {
        if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
          event.preventDefault()
          actions.saveDraft(draft, comment)
        }
      }}
      className="block w-full resize-none bg-transparent text-xs leading-5 text-ink outline-none [field-sizing:content]"
    />
  </div>
}

function NoteCard({ note, outdated, actions }: { note: ReviewNote; outdated: boolean; actions: ReviewActions }) {
  return <div className="group rounded-lg border border-divider/60 bg-surface px-3 py-2 font-sans text-xs text-ink">
    <div className="flex items-start gap-2">
      <div className="min-w-0 flex-1">
        {outdated && <p className="mb-1 text-[10px] text-ink-faint">Line {note.line} changed since this note</p>}
        {note.selectedText && <pre className="mb-1.5 max-h-32 overflow-auto whitespace-pre-wrap font-mono text-[11px] text-ink-muted">{note.selectedText}</pre>}
        <p className="whitespace-pre-wrap break-words leading-5">{note.comment}</p>
        {outdated && <pre className="mt-1.5 overflow-x-auto font-mono text-[11px] leading-4 text-ink-faint">{note.quote.join("\n")}</pre>}
      </div>
      <div className="-my-0.5 flex shrink-0 gap-0.5 opacity-0 group-hover:opacity-100 group-focus-within:opacity-100">
        <button type="button" aria-label="Edit note" onClick={() => actions.startDraft({ anchor: note, noteId: note.id, comment: note.comment })} className={ICON_BUTTON}>
          <Pencil className="h-3 w-3" />
        </button>
        <button type="button" aria-label="Delete note" onClick={() => actions.removeNote(note.id)} className={ICON_BUTTON}>
          <X className="h-3 w-3" />
        </button>
      </div>
    </div>
  </div>
}

/** Notes after one line, an open editor replaces the note it edits. */
export function Annotations({ notes, draft, showNewDraft, outdated, actions, className }: {
  notes: ReviewNote[]; draft: Draft | null; showNewDraft: boolean; outdated: boolean; actions: ReviewActions; className: string
}) {
  if (notes.length === 0 && !showNewDraft) return null
  return <div className={cn("space-y-1.5 py-1.5 pr-4", className)}>
    {notes.map((note) => <div key={note.id} data-review-note>
      {draft?.noteId === note.id ? <NoteEditor draft={draft} actions={actions} /> : <NoteCard note={note} outdated={outdated} actions={actions} />}
    </div>)}
    {showNewDraft && draft && <NoteEditor draft={draft} actions={actions} />}
  </div>
}
