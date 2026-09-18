"use client"

import dynamic from "next/dynamic"
import { useEffect, useMemo, useState } from "react"

export interface PlotFigure {
  data?: unknown[]
  layout?: Record<string, unknown>
  frames?: unknown[]
}

// Plotly touches `window` and is sizable (a few hundred KB minified even with the
// "dist-min" bundle), so it must load lazily, client-only, and only for chats that
// actually call the `plot` tool — never in the server bundle or an idle chat's payload.
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

/** Renders a figure the `plot` tool produced. All styling here is defaults only — every value
 *  sits behind a spread of `figure.layout` (or, per-axis/font/etc., merged with the model's own
 *  sub-keys winning), so a figure that already sets its own colors, fonts, or template renders
 *  unchanged. The tool's prompt and the model's expected `fig` shape are untouched; this only
 *  answers "what does an unstyled figure look like by default", the same job `plotly.io.templates`
 *  does, just matching this app's tokens instead of a stock template. */
export function PlotElement({ figure }: PlotElementProps) {
  // `getComputedStyle` below reads CSS custom properties once; toggling the `.light` class on
  // <html> doesn't re-run this component (nothing in its own props/state changed), so without
  // this the figure keeps the palette it was born with, colorway included. `themeTick` gives the
  // memo below a dependency that actually changes when the theme does.
  const [themeTick, setThemeTick] = useState(0)
  useEffect(() => {
    const observer = new MutationObserver(() => setThemeTick((t) => t + 1))
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] })
    return () => observer.disconnect()
  }, [])

  const layout = useMemo(() => {
    if (typeof window === "undefined") return figure.layout ?? {}
    const root = document.documentElement
    const style = getComputedStyle(root)
    const isLight = root.classList.contains("light")
    const ink = style.getPropertyValue("--ink").trim()
    const surfaceElevated = style.getPropertyValue("--surface-elevated").trim()
    const grid = style.getPropertyValue("--divider").trim()
    const sansFont = style.getPropertyValue("--font-sans").trim() || "system-ui, sans-serif"
    const axisDefaults = { gridcolor: grid, zerolinecolor: grid, linecolor: grid }
    return {
      margin: { t: 32, r: 16, b: 40, l: 48 },
      colorway: isLight ? LIGHT_COLORWAY : DARK_COLORWAY,
      ...figure.layout,
      autosize: true,
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      font: { color: ink, family: sansFont, size: 12, ...(figure.layout?.font as Record<string, unknown> | undefined) },
      xaxis: { ...axisDefaults, ...(figure.layout?.xaxis as Record<string, unknown> | undefined) },
      yaxis: { ...axisDefaults, ...(figure.layout?.yaxis as Record<string, unknown> | undefined) },
      legend: { bgcolor: "transparent", ...(figure.layout?.legend as Record<string, unknown> | undefined) },
      hoverlabel: {
        bgcolor: surfaceElevated,
        bordercolor: grid,
        font: { color: ink, family: sansFont },
        ...(figure.layout?.hoverlabel as Record<string, unknown> | undefined),
      },
    }
  }, [figure.layout, themeTick])

  return (
    <div className="w-full overflow-hidden rounded-lg">
      <Plot
        data={figure.data ?? []}
        layout={layout}
        frames={figure.frames}
        config={{ responsive: true, displaylogo: false, displayModeBar: "hover" }}
        useResizeHandler
        style={{ width: "100%", height: "420px" }}
      />
    </div>
  )
}
