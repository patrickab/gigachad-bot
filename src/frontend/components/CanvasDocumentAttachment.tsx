"use client"

import { useEffect, useState } from "react"
import { loadFileViewerText } from "@/lib/api"
import { HostScaleContext } from "@/contexts/HostScaleContext"
import type { PlotFigure } from "@/lib/types"
import { LaTeXMarkdown } from "./LaTeXMarkdown"
import { PlotElement } from "./PlotElement"

// A document attachment is a read-only view of a generated chat artifact — a Mermaid
// or markmap note rendered through the shared markdown renderer, or a Plotly figure
// through the shared plot renderer. Markdown is laid out at screen size like an
// Architecture Diagram. A plot instead lays out at a zoom-independent width wide enough
// to avoid overlapping labels, is CSS-scaled into its window, and collapses while a
// zoom gesture runs: Plotly then redraws at most once per settled zoom, not per frame.
type DocumentAttachmentState =
  | { status: "loading" }
  | { status: "markdown"; content: string }
  | { status: "plot"; figure: PlotFigure }
  | { status: "error"; message: string }

export function CanvasDocumentAttachment({ path, hostScale, plotFrame, collapsed }: {
  path: string
  hostScale: number
  /** Layout width (px) and CSS scale for a plot; null renders it at its window size (fullscreen). */
  plotFrame: { width: number; scale: number } | null
  collapsed: boolean
}) {
  const [state, setState] = useState<DocumentAttachmentState>({ status: "loading" })

  useEffect(() => {
    let alive = true
    setState({ status: "loading" })
    loadFileViewerText(path).then((text) => {
      if (!alive) return
      if (path.endsWith(".plot.json")) {
        try {
          const figure = JSON.parse(text)
          if (!figure || typeof figure !== "object" || Array.isArray(figure)) throw new Error("Invalid figure")
          setState({ status: "plot", figure: figure as PlotFigure })
        } catch {
          setState({ status: "error", message: "Malformed plot document" })
        }
      } else {
        setState({ status: "markdown", content: text })
      }
    }).catch(() => { if (alive) setState({ status: "error", message: "Failed to load document" }) })
    return () => { alive = false }
  }, [path])

  if (state.status === "loading") return <div className="flex h-full items-center justify-center text-[10px] text-ink-faint">Loading…</div>
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
