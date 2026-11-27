"use client"

import { loadFileViewerText } from "@/lib/backend/api"
import { HostScaleContext } from "@/contexts/HostScaleContext"
import type { PlotFigure } from "@/lib/domain/types"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"
import { PlotElement } from "@/components/content/PlotElement"
import { LoadingPlaceholder } from "@/components/ui/DocumentLoadState"
import { plotFigureOf } from "@/features/editor/useEditorView"
import { useLoaded } from "@/hooks/useLoaded"

// A document attachment is a read-only view of a generated chat artifact — a Mermaid
// or markmap note rendered through the shared markdown renderer, or a Plotly figure
// through the shared plot renderer. Markdown is laid out at screen size like an
// Architecture Diagram. A plot instead lays out at a zoom-independent width wide enough
// to avoid overlapping labels, is CSS-scaled into its window, and collapses while a
// zoom gesture runs: Plotly then redraws at most once per settled zoom, not per frame.
type DocumentAttachmentState =
  | { status: "markdown"; content: string }
  | { status: "plot"; figure: PlotFigure }
  | { status: "error"; message: string }

async function loadAttachmentDocument(path: string): Promise<DocumentAttachmentState> {
  let text: string
  try {
    text = await loadFileViewerText(path)
  } catch {
    return { status: "error", message: "Failed to load document" }
  }
  if (!path.endsWith(".plot.json")) return { status: "markdown", content: text }
  const figure = plotFigureOf(path, text)
  return figure ? { status: "plot", figure } : { status: "error", message: "Malformed plot document" }
}

export function CanvasDocumentAttachment({ path, hostScale, plotFrame, collapsed }: {
  path: string
  hostScale: number
  /** Layout width (px) and CSS scale for a plot; null renders it at its window size (fullscreen). */
  plotFrame: { width: number; scale: number } | null
  collapsed: boolean
}) {
  // A new path reads as loading until it resolves.
  const state = useLoaded(path, loadAttachmentDocument).value
  if (!state) return <LoadingPlaceholder size="fill" />
  if (state.status === "error") return <div className="flex h-full items-center justify-center px-3 text-center text-[10px] text-ink-faint">{state.message}</div>
  if (state.status === "plot") {
    if (!plotFrame) return <div className="h-full overflow-auto p-2"><PlotElement figure={state.figure} /></div>
    return (
      <div className="relative h-full overflow-hidden">
        <div
          className="overflow-auto p-2"
          style={{
            width: plotFrame.width,
            height: `${100 / plotFrame.scale}%`,
            transform: `scale(${plotFrame.scale})`,
            transformOrigin: "top left",
            visibility: collapsed ? "hidden" : undefined,
          }}
        >
          <PlotElement figure={state.figure} />
        </div>
        {collapsed && <div className="absolute inset-0 flex items-center justify-center text-[10px] text-ink-faint">Plot</div>}
      </div>
    )
  }
  // Markdown can hold a markmap, which is its own viewport and magnifies with the canvas.
  return <div className="h-full overflow-auto p-3 text-[12px]"><HostScaleContext.Provider value={hostScale}><LaTeXMarkdown content={state.content} /></HostScaleContext.Provider></div>
}
