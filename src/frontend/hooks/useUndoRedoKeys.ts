"use client"

import { useEventListener } from "@/hooks/useEventListener"
import { isEditableTarget, isModKey } from "@/lib/utils"

/** Ctrl/Cmd+Z undoes, Ctrl/Cmd+Shift+Z and Ctrl/Cmd+Y redo, while `enabled`. `skipEditable` leaves presses inside text fields alone. */
export function useUndoRedoKeys(undo: () => void, redo: () => void, enabled: boolean, skipEditable = false) {
  useEventListener("window", "keydown", (event) => {
    if (!isModKey(event) || (skipEditable && isEditableTarget(event.target))) return
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
