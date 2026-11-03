"use client"

import { useState, type MouseEvent } from "react"
import { useResetOnChange } from "@/hooks/useResetOnChange"
import { isPlotFigure, type PlotFigure } from "@/lib/domain/types"
import { isRecord } from "@/lib/utils"

export type EditorView = "edit" | "preview"

/**
 * The figure a `.json` document holds, null when it holds none.
 *
 * - `.plot.json` (the plot tool's documents): any JSON object, Plotly validates the interior
 * - Any other `.json`: only what `fig.to_json()` writes, a `data` array of traces beside a `layout` object
 */
export function plotFigureOf(path: string, content: string): PlotFigure | null {
  const isPlotTool = path.endsWith(".plot.json")
  // Cheap guards first: a large unrelated JSON file is edited per keystroke and must not be parsed each time.
  if (!isPlotTool && (!path.endsWith(".json") || !content.includes('"data"') || !content.includes('"layout"'))) return null
  try {
    const parsed: unknown = JSON.parse(content)
    if (!isPlotFigure(parsed)) return null
    if (isPlotTool) return parsed
    return Array.isArray(parsed.data) && parsed.data.every(isRecord) && isRecord(parsed.layout) ? parsed : null
  } catch {
    return null
  }
}

/** A Markdown document the diagram or mind map tool wrote: a markmap fence, or the diagram tool's header plus a mermaid fence.
 *  A note that merely contains a mermaid block is ordinary Markdown: it opens in edit and previews inline. */
export function isMarkdownArtifact(content: string): boolean {
  return /```markmap\b/i.test(content) || /^# Mermaid diagram\s+```mermaid\b/.test(content)
}

/** Plots, mind maps and diagrams open rendered, their text is the secondary view. */
export function isToolArtifact(path: string, content: string): boolean {
  return path.endsWith(".plot.json") || plotFigureOf(path, content) !== null || isMarkdownArtifact(content)
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
