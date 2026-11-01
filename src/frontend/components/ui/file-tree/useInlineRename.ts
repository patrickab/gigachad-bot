"use client"

import { useCallback, useState } from "react"

/** A row's inline rename: `editing` swaps its label for an input, `submit` rejects with the reason to show under it. */
export interface InlineRename {
  editing: boolean
  start: () => void
  stop: () => void
  submit: (name: string) => Promise<void>
  /** Select the name without its suffix, e.g. `paper` of `paper.tex`. */
  keepSuffix?: boolean
}

/**
 * Which row of a tree is being renamed: one at a time, by a `key` unique across the trees showing it.
 *
 * - `renameOf` gives a row's `FileTreeItem.rename`, `start` begins a rename from outside (F2, a menu)
 */
export function useInlineRename() {
  const [renamingKey, setRenamingKey] = useState<string | null>(null)
  const renameOf = useCallback((key: string, submit: (name: string) => Promise<void>, keepSuffix = false): InlineRename => ({
    editing: renamingKey === key,
    start: () => setRenamingKey(key),
    stop: () => setRenamingKey((current) => current === key ? null : current),
    submit,
    keepSuffix,
  }), [renamingKey])
  return { renameOf, start: setRenamingKey }
}
