type PlotLayout = Record<string, unknown>

/** App colors a plot is themed with, read from the CSS variables. */
export interface PlotTheme {
  isLight: boolean
  ink: string
  surfaceElevated: string
  grid: string
  sansFont: string
}

// A small categorical palette, not the app's monochrome ink scale: distinguishing 4-6 traces
// needs separable hues, but stays muted (low chroma) rather than the saturated rainbow Plotly
// defaults to, so it still reads as "this app" and not "a generic chart library".
const COLORWAY_HUES = [250, 25, 150, 320, 200, 60]
const DARK_COLORWAY = COLORWAY_HUES.map((h) => `oklch(72% 0.11 ${h})`)
const LIGHT_COLORWAY = COLORWAY_HUES.map((h) => `oklch(52% 0.13 ${h})`)

const MODEL_SIZING_KEYS = new Set(["autosize", "height", "width"])

/** A model layout entry spread as the model wrote it. */
const entry = (value: unknown) => value as PlotLayout | undefined

// Row count from `layout.grid.rows`, else the number of distinct yaxis domains, so stacked subplots get proportional height.
export function subplotRowsFor(layout: PlotLayout | undefined) {
  const gridRows = entry(layout?.grid)?.rows
  if (typeof gridRows === "number" && Number.isInteger(gridRows) && gridRows > 0) return gridRows

  const domains = Object.entries(layout ?? {})
    .filter(([key, value]) => /^yaxis\d*$/.test(key) && value && typeof value === "object")
    .map(([, axis]) => (axis as PlotLayout).domain)
    .filter((domain): domain is [number, number] => Array.isArray(domain)
      && domain.length === 2
      && domain.every((value) => typeof value === "number"))
    .map(([start, end]) => `${start}:${end}`)
  return Math.max(1, new Set(domains).size)
}

// Per-row height: 0.6 aspect ratio clamped to 300-560px
export function chartHeightFor(width: number, rows: number) {
  return Math.min(560, Math.max(300, Math.round(width * 0.6))) * rows
}

/** The model's layout without its sizing: sizing comes from the measured host, never from the model. */
export function withoutModelSizing(layout: PlotLayout | undefined): PlotLayout {
  return Object.fromEntries(Object.entries(layout ?? {}).filter(([key]) => !MODEL_SIZING_KEYS.has(key)))
}

/** The app theme from the document's CSS variables. */
export function readPlotTheme(isLight: boolean): PlotTheme {
  const style = getComputedStyle(document.documentElement)
  return {
    isLight,
    ink: style.getPropertyValue("--ink").trim(),
    surfaceElevated: style.getPropertyValue("--surface-elevated").trim(),
    grid: style.getPropertyValue("--chart-grid").trim(),
    sansFont: style.getPropertyValue("--font-sans").trim() || "system-ui, sans-serif",
  }
}

/** Model layout over themed defaults. Sizing, transparent backgrounds and axis colors are always themed. */
export function themedPlotLayout(modelLayout: PlotLayout, { isLight, ink, surfaceElevated, grid, sansFont }: PlotTheme, size: { width: number; height: number } | null): PlotLayout {
  const axisDefaults = { gridcolor: grid, zerolinecolor: grid, linecolor: grid }
  // WebGL composites translucent walls on a transparent scene as near-white, so 3D walls stay off
  const sceneAxisDefaults = { ...axisDefaults, showbackground: false }
  const themedAxes = Object.fromEntries(
    Object.entries(modelLayout)
      .filter(([key, value]) => /^(?:x|y)axis\d*$/.test(key) && value && typeof value === "object")
      .map(([key, value]) => [key, { ...entry(value), ...axisDefaults }]),
  )
  const modelScene = entry(modelLayout.scene)
  return {
    margin: { t: 32, r: 16, b: 40, l: 48 },
    colorway: isLight ? LIGHT_COLORWAY : DARK_COLORWAY,
    ...modelLayout,
    ...themedAxes,
    autosize: false,
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { ...entry(modelLayout.font), color: ink, family: sansFont, size: 12 },
    xaxis: { ...entry(modelLayout.xaxis), ...axisDefaults },
    yaxis: { ...entry(modelLayout.yaxis), ...axisDefaults },
    legend: { bgcolor: "transparent", ...entry(modelLayout.legend) },
    scene: {
      ...modelScene,
      aspectmode: modelScene?.aspectmode ?? "cube",
      bgcolor: "transparent",
      xaxis: { ...entry(modelScene?.xaxis), ...sceneAxisDefaults },
      yaxis: { ...entry(modelScene?.yaxis), ...sceneAxisDefaults },
      zaxis: { ...entry(modelScene?.zaxis), ...sceneAxisDefaults },
    },
    hoverlabel: {
      bgcolor: surfaceElevated,
      bordercolor: grid,
      font: { color: ink, family: sansFont },
      ...entry(modelLayout.hoverlabel),
    },
    hovermode: false,
    width: size?.width,
    height: size?.height,
  }
}
