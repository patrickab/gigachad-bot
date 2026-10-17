"use client"

import dynamic from "next/dynamic"
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Download, Maximize2, Minimize2 } from "lucide-react"
import { useTwoFingerGesture } from "@/hooks/useTwoFingerGesture"
import { cn } from "@/lib/utils"

export interface PlotFigure {
  data?: unknown[]
  layout?: Record<string, unknown>
  frames?: unknown[]
}

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

// A small categorical palette, not the app's monochrome ink scale: distinguishing 4-6 traces
// needs separable hues, but stays muted (low chroma) rather than the saturated rainbow Plotly
// defaults to, so it still reads as "this app" and not "a generic chart library".
const COLORWAY_HUES = [250, 25, 150, 320, 200, 60]
const DARK_COLORWAY = COLORWAY_HUES.map((h) => `oklch(72% 0.11 ${h})`)
const LIGHT_COLORWAY = COLORWAY_HUES.map((h) => `oklch(52% 0.13 ${h})`)

// Row count from `layout.grid.rows`, else the number of distinct yaxis domains, so stacked subplots get proportional height.
function subplotRowsFor(layout: Record<string, unknown> | undefined) {
  const gridRows = (layout?.grid as Record<string, unknown> | undefined)?.rows
  if (typeof gridRows === "number" && Number.isInteger(gridRows) && gridRows > 0) return gridRows

  const domains = Object.entries(layout ?? {})
    .filter(([key, value]) => /^yaxis\d*$/.test(key) && value && typeof value === "object")
    .map(([, axis]) => (axis as Record<string, unknown>).domain)
    .filter((domain): domain is [number, number] => Array.isArray(domain)
      && domain.length === 2
      && domain.every((value) => typeof value === "number"))
    .map(([start, end]) => `${start}:${end}`)
  return Math.max(1, new Set(domains).size)
}

// Per-row height: 0.6 aspect ratio clamped to 300-560px
function chartHeightFor(width: number, rows: number) {
  return Math.min(560, Math.max(300, Math.round(width * 0.6))) * rows
}

// react-plotly calls `Plotly.react` whenever `data`, `layout` or `config` changes identity,
// so these must stay module-level: an inline object redraws every plot on every host render.
const PLOT_CONFIG = { responsive: false, displaylogo: false, displayModeBar: false, scrollZoom: true } as const
const PINCH_WHEEL_DELTA = 200 // Plotly zooms by exp(-deltaY / 200) per wheel event
const PINCH_ZOOM_GAIN = 4.5 // zoom = fingerRatio ** gain; 1 tracks the fingers exactly but feels slow
const NO_TRACES: unknown[] = []

/** Renders a figure the `sandbox_plot` tool produced. Theme values remain defaults the model can
 *  override; dimensions and hover behavior are controlled by the chat host. Memoized: canvas
 *  pan/zoom re-renders its host every frame while the figure stays the same object. */
export const PlotElement = memo(function PlotElement({ figure }: PlotElementProps) {
  const hostRef = useRef<HTMLDivElement>(null)
  const graphRef = useRef<HTMLElement | null>(null)
  // Height only counts in fullscreen; inline it follows the width-based aspect ratio.
  const [hostSize, setHostSize] = useState({ width: 0, height: 0 })
  const [isFullscreen, setIsFullscreen] = useState(false)
  // Theme colors are read once in the layout memo below, and a `.light` class toggle on <html>
  // changes none of its deps. `themeTick` bumps on that toggle to force a re-read.
  const [themeTick, setThemeTick] = useState(0)

  useEffect(() => {
    const host = hostRef.current
    if (!host) return

    // Ignore zero widths (hidden/collapsed host) so Plot never mounts at 0px
    const updateSize = (width: number, height: number) => setHostSize((current) => {
      const next = {
        width: Math.round(width),
        height: host === document.fullscreenElement ? Math.round(height) : 0,
      }
      const isUnchanged = next.width === current.width && next.height === current.height
      return next.width > 0 && !isUnchanged ? next : current
    })
    const observer = new ResizeObserver(([entry]) => updateSize(
      entry?.contentRect.width ?? host.clientWidth,
      entry?.contentRect.height ?? host.clientHeight,
    ))
    observer.observe(host)
    updateSize(host.clientWidth, host.clientHeight)

    // Measure in the same batch as the flag so the plot redraws once at its final size.
    const syncFullscreen = () => {
      setIsFullscreen(document.fullscreenElement === host)
      updateSize(host.clientWidth, host.clientHeight)
    }
    document.addEventListener("fullscreenchange", syncFullscreen)
    return () => {
      observer.disconnect()
      document.removeEventListener("fullscreenchange", syncFullscreen)
    }
  }, [])

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

  useEffect(() => {
    const observer = new MutationObserver(() => setThemeTick((t) => t + 1))
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] })
    return () => observer.disconnect()
  }, [])

  // Pinch zooms the plot; a plain wheel scrolls the page instead (except fullscreen, which has
  // nothing to scroll). Trackpad pinches arrive as ctrl+wheel, which must not zoom the page.
  // Two-finger touches are hidden from Plotly, whose own drag would rotate or pan mid-pinch.
  useEffect(() => {
    const host = hostRef.current
    if (!host) return
    const onWheel = (event: WheelEvent) => {
      if (event.ctrlKey) event.preventDefault()
      else if (document.fullscreenElement !== host) event.stopPropagation()
    }
    const onMultiTouch = (event: TouchEvent) => {
      if (event.touches.length >= 2) event.stopPropagation()
    }
    host.addEventListener("wheel", onWheel, { capture: true, passive: false })
    host.addEventListener("touchstart", onMultiTouch, { capture: true })
    host.addEventListener("touchmove", onMultiTouch, { capture: true })
    return () => {
      host.removeEventListener("wheel", onWheel, { capture: true })
      host.removeEventListener("touchstart", onMultiTouch, { capture: true })
      host.removeEventListener("touchmove", onMultiTouch, { capture: true })
    }
  }, [])

  // Touch pinch has no Plotly equivalent, so replay it as ctrl+wheel, which cartesian, 3D and geo
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
        bubbles: true, cancelable: true, clientX, clientY, ctrlKey: true, deltaY: -PINCH_WHEEL_DELTA * PINCH_ZOOM_GAIN * Math.log(stepRatio),
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
    const { autosize: _autosize, height: _height, width: _width, ...modelLayout } = figure.layout ?? {}
    if (typeof window === "undefined") return modelLayout
    const root = document.documentElement
    const style = getComputedStyle(root)
    const isLight = root.classList.contains("light")
    const ink = style.getPropertyValue("--ink").trim()
    const surfaceElevated = style.getPropertyValue("--surface-elevated").trim()
    const grid = style.getPropertyValue("--chart-grid").trim()
    const sansFont = style.getPropertyValue("--font-sans").trim() || "system-ui, sans-serif"
    const axisDefaults = { gridcolor: grid, zerolinecolor: grid, linecolor: grid }
    // WebGL composites translucent walls on a transparent scene as near-white, so 3D walls stay off
    const sceneAxisDefaults = { ...axisDefaults, showbackground: false }
    const themedAxes = Object.fromEntries(
      Object.entries(modelLayout)
        .filter(([key, value]) => /^(?:x|y)axis\d*$/.test(key) && value && typeof value === "object")
        .map(([key, value]) => [key, { ...(value as Record<string, unknown>), ...axisDefaults }]),
    )
    const modelScene = modelLayout.scene as Record<string, unknown> | undefined
    // Model layout wins over defaults. Sizing, transparent backgrounds and axis colors are always themed.
    return {
      margin: { t: 32, r: 16, b: 40, l: 48 },
      colorway: isLight ? LIGHT_COLORWAY : DARK_COLORWAY,
      ...modelLayout,
      ...themedAxes,
      autosize: false,
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      font: { ...(modelLayout.font as Record<string, unknown> | undefined), color: ink, family: sansFont, size: 12 },
      xaxis: { ...(modelLayout.xaxis as Record<string, unknown> | undefined), ...axisDefaults },
      yaxis: { ...(modelLayout.yaxis as Record<string, unknown> | undefined), ...axisDefaults },
      legend: { bgcolor: "transparent", ...(modelLayout.legend as Record<string, unknown> | undefined) },
      scene: {
        ...modelScene,
        aspectmode: modelScene?.aspectmode ?? "cube",
        bgcolor: "transparent",
        xaxis: { ...(modelScene?.xaxis as Record<string, unknown> | undefined), ...sceneAxisDefaults },
        yaxis: { ...(modelScene?.yaxis as Record<string, unknown> | undefined), ...sceneAxisDefaults },
        zaxis: { ...(modelScene?.zaxis as Record<string, unknown> | undefined), ...sceneAxisDefaults },
      },
      hoverlabel: {
        bgcolor: surfaceElevated,
        bordercolor: grid,
        font: { color: ink, family: sansFont },
        ...(modelLayout.hoverlabel as Record<string, unknown> | undefined),
      },
      hovermode: false,
      width: dimensions?.width,
      height: dimensions?.height,
    }
  }, [dimensions, figure.layout, themeTick])

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
