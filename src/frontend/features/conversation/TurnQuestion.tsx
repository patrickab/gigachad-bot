"use client"

import { useRef, useState, type ElementType, type ReactNode, type Ref } from "react"
import { createPortal } from "react-dom"
import { Pencil, Trash2 } from "lucide-react"
import { CopyAction, ForkAction, TurnAction, TurnRegion } from "@gigachad/chat-ui"
import { Dialog } from "@/components/ui/Dialog"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"

const QUIET_BUTTON_CLASS = "rounded-full px-3 py-1.5 text-xs text-ink-muted transition-colors hover:bg-hover hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"
const CONFIRM_BUTTON_CLASS = "rounded-full border border-divider bg-surface-elevated px-3 py-1.5 text-xs font-medium text-danger transition-colors focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-2 focus-visible:outline-ink-muted"

function turnCount(count: number): string {
  return count === 1 ? "1 later turn" : `${count} later turns`
}

/**
 * A yes/no question in the shared `Dialog`: Enter confirms (unless on Cancel), Escape and the backdrop cancel.
 * In a portal: a turn card's `backdrop-filter` would hold a fixed overlay inside the card.
 */
function ConfirmDialog({ open, icon, title, body, confirmLabel, onConfirm, onCancel }: {
  open: boolean
  icon: ElementType
  title: string
  body: string
  confirmLabel: string
  onConfirm: () => void
  onCancel: () => void
}) {
  const confirmRef = useRef<HTMLButtonElement>(null)
  if (typeof document === "undefined") return null
  return createPortal(
    <Dialog open={open} onClose={onCancel} icon={icon} title={title} initialFocusRef={confirmRef}>
      <div
        className="px-5 pb-5"
        onKeyDown={(event) => {
          if (event.key !== "Enter" || (event.target as HTMLElement).dataset.cancel !== undefined) return
          event.preventDefault()
          onConfirm()
        }}
      >
        <p className="text-sm text-ink-muted">{body}</p>
        <div className="flex items-center justify-end gap-2 pt-4">
          <button type="button" data-cancel onClick={onCancel} className={QUIET_BUTTON_CLASS}>Cancel</button>
          <button type="button" ref={confirmRef} onClick={onConfirm} className={CONFIRM_BUTTON_CLASS}>{confirmLabel}</button>
        </div>
      </div>
    </Dialog>,
    document.body,
  )
}

/** The prompt in a textarea, focused with the caret at its end: Enter submits, Shift+Enter is a newline, Escape cancels. */
function QuestionEditor({ initial, paused, onSubmit, onCancel }: { initial: string; paused: boolean; onSubmit: (text: string) => void; onCancel: () => void }) {
  const [draft, setDraft] = useState(initial)
  // Below an open confirmation, whose own layer takes Escape
  useEscapeLayer(onCancel, !paused)
  return (
    <textarea
      aria-label="Edit question"
      autoFocus
      value={draft}
      onFocus={(event) => event.currentTarget.setSelectionRange(event.currentTarget.value.length, event.currentTarget.value.length)}
      onChange={(event) => setDraft(event.target.value)}
      onKeyDown={(event) => {
        if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) return
        event.preventDefault()
        onSubmit(draft)
      }}
      className="block w-full resize-none rounded-md bg-transparent text-sm leading-6 text-ink outline-none ring-1 ring-divider focus-visible:ring-divider-strong [field-sizing:content]"
    />
  )
}

/**
 * A turn's question, the same in the chat and in Pi: its content, and its actions while the turn is settled.
 *
 * - `live`: no actions. Otherwise copy (when there is `text`), edit with `onEdit`, delete with `onDelete`: a read-only column
 *   passes neither and gets copy only
 * - Edit turns the question into a textarea holding `text`, focused. Enter confirms, Escape cancels, Shift+Enter is a newline;
 *   an unchanged or empty text cancels. With `laterTurns` the edit asks first, naming how many later turns go with it
 * - Delete asks first
 * - The editor's draft lives in this component, which the column keys by the turn's key: a deleted turn's draft goes with it
 */
export function TurnQuestion({ text, live = false, onEdit, onDelete, laterTurns = 0, icon, onCollapse, collapseRef, children }: {
  /** The prompt: what copy copies and the editor starts from. */
  text: string
  live?: boolean
  /** Runs once the edit is confirmed, with the new text. */
  onEdit?: (text: string) => void
  /** Runs once the delete is confirmed. */
  onDelete?: () => void
  /** Turns after this one, removed by an edit. */
  laterTurns?: number
  icon?: ReactNode
  onCollapse?: () => void
  collapseRef?: Ref<HTMLButtonElement>
  children: ReactNode
}) {
  const [editing, setEditing] = useState(false)
  const [pendingEdit, setPendingEdit] = useState<string | null>(null)
  const [confirmingDelete, setConfirmingDelete] = useState(false)

  const submit = (draft: string) => {
    if (!draft.trim() || draft.trim() === text.trim()) return setEditing(false)
    if (laterTurns > 0) return setPendingEdit(draft)
    setEditing(false)
    onEdit?.(draft)
  }
  const confirmEdit = () => {
    const draft = pendingEdit
    setPendingEdit(null)
    setEditing(false)
    if (draft !== null) onEdit?.(draft)
  }
  const confirmDelete = () => {
    setConfirmingDelete(false)
    onDelete?.()
  }

  const actions = live || editing ? [] : [
    text && <CopyAction key="copy" text={text} label="Copy message" />,
    onEdit && <TurnAction key="edit" label="Edit message" icon={<Pencil size={14} aria-hidden="true" />} onClick={() => setEditing(true)} />,
    onDelete && <TurnAction key="delete" tone="danger" label="Delete turn" icon={<Trash2 size={14} aria-hidden="true" />} onClick={() => setConfirmingDelete(true)} />,
  ].filter(Boolean)

  return (
    <>
      <TurnRegion role="question" icon={icon} onCollapse={onCollapse} collapseRef={collapseRef} actions={actions.length > 0 ? actions : undefined}>
        {editing && onEdit ? <QuestionEditor initial={text} paused={pendingEdit !== null} onSubmit={submit} onCancel={() => setEditing(false)} /> : children}
      </TurnRegion>
      {onEdit && (
        <ConfirmDialog
          open={pendingEdit !== null}
          icon={Pencil}
          title="Edit this turn?"
          body={`The edit also removes the ${turnCount(laterTurns)}.`}
          confirmLabel="Edit"
          onConfirm={confirmEdit}
          onCancel={() => setPendingEdit(null)}
        />
      )}
      {onDelete && (
        <ConfirmDialog
          open={confirmingDelete}
          icon={Trash2}
          title="Delete this turn?"
          body="Its question and answer are deleted for good."
          confirmLabel="Delete"
          onConfirm={confirmDelete}
          onCancel={() => setConfirmingDelete(false)}
        />
      )}
    </>
  )
}

/**
 * A turn's answer region with its actions while the turn is settled: copy (when there is `text`) and branch with `onBranch`.
 * A live turn has none.
 */
export function TurnAnswer({ text, live = false, onBranch, branchLabel, tail, children }: {
  /** What copy copies. */
  text: string
  live?: boolean
  onBranch?: () => void
  /** The branch action's name, e.g. "Branch session from this point". */
  branchLabel: string
  tail?: "text" | "activity"
  children: ReactNode
}) {
  const actions = live ? [] : [
    text && <CopyAction key="copy" text={text} label="Copy message" />,
    onBranch && <ForkAction key="branch" label={branchLabel} onFork={onBranch} />,
  ].filter(Boolean)
  return (
    <TurnRegion role="answer" tail={tail} actions={actions.length > 0 ? actions : undefined}>
      {children}
    </TurnRegion>
  )
}
