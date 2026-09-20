"use client"

import dynamic from "next/dynamic"
import { useEffect, useMemo, useRef, useState } from "react"

export interface PlotFigure {
  data?: unknown[]
  layout?: Record<string, unknown>
  frames?: unknown[]
}

// Plotly touches `window` and is sizable (a few hundred KB minified even with the
// "dist-min" bundle), so it must load lazily, client-only, and only for chats that
// actually call the `sandbox_plot` tool — never in the server bundle or an idle chat's payload.
const Plot = dynamic(
  async () => {
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

function chartHeightFor(width: number, rows: number) {
  return Math.min(560, Math.max(300, Math.round(width * 0.6))) * rows
}

/** Renders a figure the `sandbox_plot` tool produced. Theme values remain defaults the model can
 *  override; dimensions and hover behavior are controlled by the chat host. */
export function PlotElement({ figure }: PlotElementProps) {
  const hostRef = useRef<HTMLDivElement>(null)
  const [hostWidth, setHostWidth] = useState(0)
  // `getComputedStyle` below reads CSS custom properties once; toggling the `.light` class on
  // <html> doesn't re-run this component (nothing in its own props/state changed), so without
  // this the figure keeps the palette it was born with, colorway included. `themeTick` gives the
  // memo below a dependency that actually changes when the theme does.
  const [themeTick, setThemeTick] = useState(0)

  useEffect(() => {
    const host = hostRef.current
    if (!host) return

    const updateWidth = (width: number) => setHostWidth((current) => {
      const next = Math.round(width)
      return next > 0 && next !== current ? next : current
    })
    const observer = new ResizeObserver(([entry]) => updateWidth(entry?.contentRect.width ?? host.clientWidth))
    observer.observe(host)
    updateWidth(host.clientWidth)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    const observer = new MutationObserver(() => setThemeTick((t) => t + 1))
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] })
    return () => observer.disconnect()
  }, [])

  const subplotRows = useMemo(() => subplotRowsFor(figure.layout), [figure.layout])
  const dimensions = useMemo(
    () => hostWidth > 0 ? { width: hostWidth, height: chartHeightFor(hostWidth, subplotRows) } : null,
    [hostWidth, subplotRows],
  )

  const layout = useMemo(() => {
    const { autosize: _autosize, height: _height, width: _width, ...modelLayout } = figure.layout ?? {}
    if (typeof window === "undefined") return modelLayout
    const root = document.documentElement
    const style = getComputedStyle(root)
    const isLight = root.classList.contains("light")
    const ink = style.getPropertyValue("--ink").trim()
    const surfaceElevated = style.getPropertyValue("--surface-elevated").trim()
    const grid = style.getPropertyValue("--divider").trim()
    const sansFont = style.getPropertyValue("--font-sans").trim() || "system-ui, sans-serif"
    const axisDefaults = { gridcolor: grid, zerolinecolor: grid, linecolor: grid }
    const themedAxes = Object.fromEntries(
      Object.entries(modelLayout)
        .filter(([key, value]) => /^(?:x|y)axis\d*$/.test(key) && value && typeof value === "object")
        .map(([key, value]) => [key, { ...(value as Record<string, unknown>), ...axisDefaults }]),
    )
    const modelScene = modelLayout.scene as Record<string, unknown> | undefined
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
        xaxis: { ...(modelScene?.xaxis as Record<string, unknown> | undefined), ...axisDefaults },
        yaxis: { ...(modelScene?.yaxis as Record<string, unknown> | undefined), ...axisDefaults },
        zaxis: { ...(modelScene?.zaxis as Record<string, unknown> | undefined), ...axisDefaults },
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
    <div ref={hostRef} className="w-full overflow-hidden rounded-lg">
      {dimensions && (
        <Plot
          data={figure.data ?? []}
          layout={layout}
          frames={figure.frames}
          config={{ responsive: false, displaylogo: false, displayModeBar: "hover" }}
          style={{ width: "100%", height: `${dimensions.height}px` }}
        />
      )}
    </div>
  )
}
