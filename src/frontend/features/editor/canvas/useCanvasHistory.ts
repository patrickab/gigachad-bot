"use client"

import { useCallback, useLayoutEffect, useRef, useState } from "react"
import { useEventListener } from "@/hooks/useEventListener"
import { historyChanges, restoreHistory, type CanvasDocument } from "@/lib/domain/canvasDocument"
import { isEditableTarget } from "@/lib/utils"

const HISTORY_LIMIT = 50 // undo depth

interface CanvasHistoryEntry {
  before: CanvasDocument
  after: CanvasDocument | null // null until the gesture's first write
}

/**
 * Undo/redo for one canvas, one entry per local gesture: the doc before it and its latest write.
 * Travelling diffs the two and replays only that delta against the live document, so remote
 * entities (and later remote edits) stay intact — and a drag costs nothing extra per pointermove.
 * `undo` and `redo` return whether they travelled.
 */
export function useCanvasHistory(doc: CanvasDocument, onChange: (doc: CanvasDocument) => void) {
  // latest doc/onChange for long-lived window listeners and for history
  const liveRef = useRef({ doc, onChange })
  useLayoutEffect(() => { liveRef.current = { doc, onChange } })

  const [undoStack, setUndoStack] = useState<CanvasHistoryEntry[]>([])
  const [redoStack, setRedoStack] = useState<CanvasHistoryEntry[]>([])
  const gestureRef = useRef<CanvasHistoryEntry | null>(null)

  // Opens a gesture. `before` must be read here, not inside a setState updater — updaters
  // run during the next render, by which point liveRef already holds the post-change doc.
  const snapshot = useCallback(() => {
    gestureRef.current = { before: liveRef.current.doc, after: null }
    setRedoStack([])
  }, [])

  // Writing through this keeps `liveRef` in step within the same tick: two mutations can
  // land back to back before React re-renders (a cross-canvas stroke drop removes here
  // and adds there), and the second must not read the pre-first document.
  const applyChange = useCallback((next: CanvasDocument) => {
    const gesture = gestureRef.current
    if (gesture) {
      // the gesture's first write puts it on the stack; later writes only move its end
      if (!gesture.after) setUndoStack((stack) => [...stack, gesture].slice(-HISTORY_LIMIT))
      gesture.after = next
    }
    liveRef.current.doc = next
    liveRef.current.onChange(next)
  }, [])

  // Discrete (non-drag) mutation: snapshot, then apply.
  const commit = useCallback((next: CanvasDocument) => {
    snapshot()
    applyChange(next)
    gestureRef.current = null
  }, [snapshot, applyChange])

  // A drag fires onChange on every pointermove, so it must snapshot only once — and
  // only on the move that actually changes something, so a click that moves nothing
  // (or an eraser pass over empty space) leaves no dead undo step. Reset by `endGesture`.
  const gestureSnapped = useRef(false)
  const snapshotOnce = useCallback(() => {
    if (gestureSnapped.current) return
    gestureSnapped.current = true
    snapshot()
  }, [snapshot])

  /** Close the open gesture; the next write starts a new undo step. */
  const endGesture = useCallback(() => {
    gestureRef.current = null
    gestureSnapped.current = false
  }, [])

  // Returns whether anything was travelled, so callers can react (e.g. drop a selection the restore may have invalidated).
  const travel = useCallback((direction: "undo" | "redo"): boolean => {
    const stack = direction === "undo" ? undoStack : redoStack
    const entry = stack[stack.length - 1]
    if (!entry?.after) return false
    gestureRef.current = null
    const current = liveRef.current.doc
    const restored = restoreHistory(current, historyChanges(entry.before, entry.after), direction)
    if (restored !== current) applyChange(restored)
    if (direction === "undo") {
      setUndoStack((stack) => stack.slice(0, -1))
      setRedoStack((stack) => [...stack, entry])
    } else {
      setRedoStack((stack) => stack.slice(0, -1))
      setUndoStack((stack) => [...stack, entry].slice(-HISTORY_LIMIT))
    }
    return true
  }, [undoStack, redoStack, applyChange])

  const undo = useCallback(() => travel("undo"), [travel])
  const redo = useCallback(() => travel("redo"), [travel])

  return { liveRef, applyChange, commit, snapshot, snapshotOnce, endGesture, undo, redo, canUndo: undoStack.length > 0, canRedo: redoStack.length > 0 }
}

/** Ctrl/Cmd+Z undoes, Ctrl/Cmd+Shift+Z and Ctrl/Cmd+Y redo, while `enabled`. `skipEditable` leaves presses inside text fields alone. */
export function useUndoRedoKeys(undo: () => void, redo: () => void, enabled: boolean, skipEditable = false) {
  useEventListener("window", "keydown", (event) => {
    if (!(event.ctrlKey || event.metaKey) || (skipEditable && isEditableTarget(event.target))) return
    const key = event.key.toLowerCase() // Shift uppercases it, so Ctrl+Shift+Z arrives as "Z"
    // Redo also matches `code` — the physical key — for layouts where `key` isn't "y".
    // The undo branch must stay first: on QWERTZ the KeyY position is labelled Z.
    if (key === "z" && !event.shiftKey) {
      event.preventDefault()
      undo()
    } else if (key === "y" || event.code === "KeyY" || (key === "z" && event.shiftKey)) {
      event.preventDefault()
      redo()
    }
  }, { active: enabled })
}
