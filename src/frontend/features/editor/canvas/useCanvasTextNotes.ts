"use client"

import { useCallback, useRef, useState, type KeyboardEvent as ReactKeyboardEvent } from "react"
import { TEXT_DEFAULT_HEIGHT, TEXT_DEFAULT_WIDTH, canvasEntityId, type CanvasDocument } from "@/lib/domain/canvasDocument"
import { isEditableTarget, isModKey } from "@/lib/utils"

const TEXT_DEFAULT_SIZE = 28 // canvas units — font size for new text notes

interface CanvasTextNotesOptions {
  liveRef: { readonly current: { doc: CanvasDocument } }
  snapshot: () => void
  applyChange: (next: CanvasDocument) => void
  endGesture: () => void
  /** Ink colour a new note is written in. */
  penColor: string
  pointAtCenter: (width: number, aspect: number) => { cx: number; cy: number }
}

/**
 * Text notes: typing on the bare canvas starts one at the pointer. Notes render as Markdown when
 * unfocused; Enter switches from source to formatted content, clicking the rendered note reopens its textarea.
 */
export function useCanvasTextNotes({ liveRef, snapshot, applyChange, endGesture, penColor, pointAtCenter }: CanvasTextNotesOptions) {
  const pointerPointRef = useRef<[number, number] | null>(null)
  const newTextIds = useRef(new Set<string>())
  const [autoFocusId, setAutoFocusId] = useState<string | null>(null)
  const [editingTextId, setEditingTextId] = useState<string | null>(null)
  // One history gesture per focus session of a text note, closed on blur, so typing a
  // sentence is one undo step, not one per keystroke. A freshly created box opens no new
  // one — its creation gesture (from addText) stays open and covers the typing.
  const textEdited = useRef(false)

  const addText = useCallback((x: number, y: number, text: string) => {
    const id = canvasEntityId()
    newTextIds.current.add(id)
    const cur = liveRef.current.doc
    // The gesture stays open until the note blurs, so what is typed into it undoes together with its creation.
    snapshot()
    applyChange({ ...cur, texts: [...cur.texts, { id, x, y, width: TEXT_DEFAULT_WIDTH, height: TEXT_DEFAULT_HEIGHT, text, color: penColor, size: TEXT_DEFAULT_SIZE }] })
    setAutoFocusId(id)
    setEditingTextId(id)
  }, [liveRef, penColor, snapshot, applyChange])

  const startEditing = useCallback((id: string) => {
    setEditingTextId(id)
    setAutoFocusId(id)
  }, [])

  const editText = useCallback((id: string, text: string) => {
    const cur = liveRef.current.doc
    if (!textEdited.current) {
      textEdited.current = true
      if (cur.texts.find((t) => t.id === id)?.text !== "") snapshot()
    }
    applyChange({ ...cur, texts: cur.texts.map((t) => t.id === id ? { ...t, text } : t) })
  }, [liveRef, snapshot, applyChange])

  const focusText = useCallback((id: string) => {
    textEdited.current = newTextIds.current.delete(id)
    setAutoFocusId((cur) => cur === id ? null : cur)
  }, [])

  // Removes a text note if its content is blank — called on blur so an
  // untouched or fully-cleared box doesn't linger as an empty artifact.
  const blurText = useCallback((id: string) => {
    setEditingTextId(null)
    const cur = liveRef.current.doc
    const item = cur.texts.find((t) => t.id === id)
    if (item && item.text.trim() === "") {
      applyChange({ ...cur, texts: cur.texts.filter((t) => t.id !== id) })
    }
    endGesture()
  }, [liveRef, applyChange, endGesture])

  const handleCanvasKeyDown = useCallback((e: ReactKeyboardEvent<HTMLDivElement>) => {
    // Only a single printable character starts a note (named keys like "Enter" are longer).
    if (isModKey(e) || e.altKey || e.nativeEvent.isComposing || Array.from(e.key).length !== 1) return
    // Only the bare canvas surface should turn typing into a new text note.
    // Any nested editable (architecture diagram fields, modals, contenteditable) keeps its keystrokes.
    if (isEditableTarget(e.target, { includeSelect: true })) return

    e.preventDefault()
    const centered = pointAtCenter(TEXT_DEFAULT_WIDTH, TEXT_DEFAULT_HEIGHT / TEXT_DEFAULT_WIDTH)
    const [x, y] = pointerPointRef.current ?? [centered.cx, centered.cy]
    addText(x, y, e.key)
  }, [addText, pointAtCenter])

  return { pointerPointRef, autoFocusId, editingTextId, startEditing, editText, focusText, blurText, handleCanvasKeyDown }
}
