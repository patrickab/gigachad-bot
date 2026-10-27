"use client"

import { useState } from "react"
import { useResetOnChange } from "@/hooks/useResetOnChange"

export type EditorView = "edit" | "preview"

/** Plots and mind maps open rendered, their text is the secondary view. */
export function isToolArtifact(path: string, content: string): boolean {
  return path.endsWith(".plot.json") || /```markmap\b/i.test(content)
}

/** Edit or preview mode, defaulted once per document when its content first loads: preview for tool artifacts. */
export function useEditorView(path: string, content: string | null, loadedPath: string | null) {
  const [view, setView] = useState<EditorView>(() => path.endsWith(".plot.json") ? "preview" : "edit")
  const readyPath = content !== null && loadedPath === path ? path : null
  useResetOnChange(readyPath, (ready) => {
    if (ready !== null) setView(isToolArtifact(ready, content!) ? "preview" : "edit")
  })
  return [view, setView] as const
}
