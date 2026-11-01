"use client"

import { useState, type MouseEvent } from "react"
import { useResetOnChange } from "@/hooks/useResetOnChange"

export type EditorView = "edit" | "preview"

/** Plots, mind maps and the diagram tool's documents open rendered, their text is the secondary view.
 *  A note that merely contains a mermaid block is ordinary Markdown: it opens in edit and previews inline. */
export function isToolArtifact(path: string, content: string): boolean {
  return path.endsWith(".plot.json") || /```markmap\b/i.test(content) || /^# Mermaid diagram\s+```mermaid\b/.test(content)
}

/** Classes that let a tool artifact's figure take its container: `cqh` is the container's height, which the map and diagram read as their own, instead of the inline chat size. */
export const ARTIFACT_FILL = "[container-type:size] [--markmap-height:100cqh] [&_.gc-diagram_svg]:h-[calc(100cqh-5rem)]! [&_.gc-diagram_svg]:max-w-none! [&_.gc-diagram_svg]:w-full"

/** A click on a rendered `#file:<path>` link hands `<path>` to `onNavigate` instead of changing the URL hash. */
export function followFileLink(event: MouseEvent, onNavigate: ((path: string) => void) | undefined): void {
  const href = (event.target as HTMLElement).closest("a")?.getAttribute("href")
  if (!href?.startsWith("#file:")) return
  event.preventDefault()
  onNavigate?.(decodeURIComponent(href.slice("#file:".length)))
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
