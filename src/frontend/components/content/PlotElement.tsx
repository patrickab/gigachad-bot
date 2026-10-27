"use client"

import dynamic from "next/dynamic"
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Download, Maximize2, Minimize2 } from "lucide-react"
import { useEventListener } from "@/hooks/useEventListener"
import { useThemeName } from "@/hooks/useThemeName"
import { useTwoFingerGesture } from "@/hooks/useTwoFingerGesture"
import type { PlotFigure } from "@/lib/domain/types"
import { chartHeightFor, readPlotTheme, subplotRowsFor, themedPlotLayout, withoutModelSizing } from "./plotTheme"

// Plotly touches `window` and is sizable (a few hundred KB minified even with the
// "dist-min" bundle), so it must load lazily, client-only, and only for chats that
// actually call the `sandbox_plot` tool, never in the server bundle or an idle chat's payload.
const Plot = dynamic(
  async () => {
    // Plotly detects the global at initialization, so MathJax must load first.
    await import("mathjax/es5/tex-svg.js")
    const [{ default: createPlotlyComponent }, { default: Plotly }] = await Promise.all([
      import("react-plotly.js/factory"),
      import("plotly.js-dist-min"),
    ])
    return createPlotlyComponent(Plotly)
  },
  {
    ssr: false,
    loading: () => <div className="flex h-64 items-center justify-center text-xs text-ink-faint">Loading plot…</div>,
  }
)

interface PlotElementProps {
  figure: PlotFigure
}

// react-plotly calls `Plotly.react` whenever `data`, `layout` or `config` changes identity,
// so these must stay module-level: an inline object redraws every plot on every host render.
const PLOT_CONFIG = { responsive: false, displaylogo: false, displayModeBar: false, scrollZoom: true } as const
const PINCH_WHEEL_DELTA = 200 // Plotly zooms by exp(-deltaY / 200) per wheel event
const PINCH_ZOOM_GAIN = 4.5 // zoom = fingerRatio ** gain; 1 tracks the fingers exactly but feels slow
const NO_TRACES: unknown[] = []

/** Host size updater: the width, plus the height in fullscreen. Zero widths (hidden/collapsed host) are ignored so Plot never mounts at 0px. */
const sizeUpdate = (host: HTMLElement, width: number, height: number) => (current: { width: number; height: number }) => {
  const next = {
    width: Math.round(width),
    height: host === document.fullscreenElement ? Math.round(height) : 0,
  }
  const isUnchanged = next.width === current.width && next.height === current.height
  return next.width > 0 && !isUnchanged ? next : current
}

/** Renders a figure the `sandbox_plot` tool produced. Theme values remain defaults the model can
 *  override; dimensions and hover behavior are controlled by the chat host. Memoized: canvas
 *  pan/zoom re-renders its host every frame while the figure stays the same object. */
export const PlotElement = memo(function PlotElement({ figure }: PlotElementProps) {
  const hostRef = useRef<HTMLDivElement>(null)
  const graphRef = useRef<HTMLElement | null>(null)
  // Height only counts in fullscreen; inline it follows the width-based aspect ratio.
  const [hostSize, setHostSize] = useState({ width: 0, height: 0 })
  const [isFullscreen, setIsFullscreen] = useState(false)
  // Theme colors are read once in the layout memo below, so a theme flip must re-run it.
  const themeName = useThemeName()

  useEffect(() => {
    const host = hostRef.current
    if (!host) return
    const observer = new ResizeObserver(([entry]) => setHostSize(sizeUpdate(
      host,
      entry?.contentRect.width ?? host.clientWidth,
      entry?.contentRect.height ?? host.clientHeight,
    )))
    observer.observe(host)
    setHostSize(sizeUpdate(host, host.clientWidth, host.clientHeight))
    return () => observer.disconnect()
  }, [])

  // Measure in the same batch as the flag so the plot redraws once at its final size.
  useEventListener("document", "fullscreenchange", () => {
    const host = hostRef.current
    if (!host) return
    setIsFullscreen(document.fullscreenElement === host)
    setHostSize(sizeUpdate(host, host.clientWidth, host.clientHeight))
  })

  const toggleFullscreen = useCallback(() => {
    if (document.fullscreenElement === hostRef.current) void document.exitFullscreen()
    else void hostRef.current?.requestFullscreen()
  }, [])

  /** Export a PNG on the page background, since the live plot is transparent. */
  const downloadPng = useCallback(async () => {
    const graph = graphRef.current
    if (!graph) return
    const { default: Plotly } = await import("plotly.js-dist-min")
    const paper = getComputedStyle(document.documentElement).getPropertyValue("--paper").trim()
    await Plotly.relayout(graph, { paper_bgcolor: paper })
    try {
      await Plotly.downloadImage(graph, { format: "png", filename: "plot", width: graph.clientWidth, height: graph.clientHeight })
    } finally {
      await Plotly.relayout(graph, { paper_bgcolor: "transparent" })
    }
  }, [])

  // A scroll gesture zooms the plot, inline and fullscreen alike. Trackpad pinches arrive as trusted
  // ctrl+wheel events, which Plotly zooms far too slowly and the browser would turn into page zoom,
  // so both are swallowed: pinch is a touchscreen gesture only.
  // Two-finger touches are hidden from Plotly, whose own drag would rotate or pan mid-pinch.
  useEventListener(hostRef, "wheel", (event) => {
    if (!event.ctrlKey || !event.isTrusted) return
    event.preventDefault()
    event.stopPropagation()
  }, { capture: true, passive: false })
  const hideMultiTouch = (event: TouchEvent) => {
    if (event.touches.length >= 2) event.stopPropagation()
  }
  useEventListener(hostRef, "touchstart", hideMultiTouch, { capture: true })
  useEventListener(hostRef, "touchmove", hideMultiTouch, { capture: true })

  // Touch pinch has no Plotly equivalent, so replay it as a wheel, which cartesian, 3D and geo
  // plots all zoom around the cursor on.
  useTwoFingerGesture(hostRef, {
    begin: () => ({ previousRatio: 1 }),
    update: (gesture, { ratio, mx, my }) => {
      const host = hostRef.current
      if (!host) return
      const stepRatio = ratio / gesture.previousRatio
      gesture.previousRatio = ratio
      const { left, top } = host.getBoundingClientRect()
      const [clientX, clientY] = [left + mx, top + my]
      const target = document.elementFromPoint(clientX, clientY) ?? host
      target.dispatchEvent(new WheelEvent("wheel", {
        bubbles: true, cancelable: true, clientX, clientY, deltaY: -PINCH_WHEEL_DELTA * PINCH_ZOOM_GAIN * Math.log(stepRatio),
      }))
    },
  })

  const subplotRows = useMemo(() => subplotRowsFor(figure.layout), [figure.layout])
  const dimensions = useMemo(
    () => hostSize.width > 0
      ? { width: hostSize.width, height: hostSize.height || chartHeightFor(hostSize.width, subplotRows) }
      : null,
    [hostSize, subplotRows],
  )

  const layout = useMemo(() => {
    const modelLayout = withoutModelSizing(figure.layout)
    if (typeof window === "undefined") return modelLayout
    return themedPlotLayout(modelLayout, readPlotTheme(themeName === "light"), dimensions)
  }, [dimensions, figure.layout, themeName])

  return (
    <div
      ref={hostRef}
      className="group relative w-full overflow-hidden rounded-lg [&:fullscreen]:rounded-none [&:fullscreen]:bg-[var(--paper)]"
    >
      {dimensions && (
        <Plot
          data={figure.data ?? NO_TRACES}
          layout={layout}
          frames={figure.frames}
          config={PLOT_CONFIG}
          style={{ width: "100%", height: `${dimensions.height}px`, touchAction: "none" }}
          onInitialized={(_figure, graph) => { graphRef.current = graph }}
        />
      )}
      <div className="absolute right-2 top-2 z-10 flex gap-1 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100 [@media(hover:none)]:opacity-100">
        <button
          type="button"
          onClick={() => void downloadPng()}
          aria-label="Download plot"
          className="rounded p-1.5 text-ink-subtle transition-colors hover:bg-hover hover:text-ink"
        >
          <Download className="h-3.5 w-3.5" aria-hidden="true" />
        </button>
        <button
          type="button"
          onClick={toggleFullscreen}
          aria-label={isFullscreen ? "Exit plot fullscreen" : "Enter plot fullscreen"}
          className="rounded p-1.5 text-ink-subtle transition-colors hover:bg-hover hover:text-ink"
        >
          {isFullscreen ? <Minimize2 className="h-3.5 w-3.5" aria-hidden="true" /> : <Maximize2 className="h-3.5 w-3.5" aria-hidden="true" />}
        </button>
      </div>
    </div>
  )
})
