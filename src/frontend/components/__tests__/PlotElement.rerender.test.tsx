/**
 * Plots on a canvas sit under a host that re-renders on every pan/draw pointer event.
 * react-plotly redraws (`Plotly.react`) whenever data/layout/config change identity, so a
 * host re-render with the same figure must not reach Plotly. Uses the real react-plotly
 * factory (separate file: PlotElement.test.tsx stubs the component out entirely).
 */
import { act, render } from "@testing-library/react"
import { useState } from "react"
import { expect, it, vi } from "vitest"

const reactCalls = vi.hoisted(() => ({ n: 0 }))

vi.mock("next/dynamic", async () => {
  // Inside vi.mock's factory a static import is unavailable; the module is fixed.
  const { default: createPlotlyComponent } = await import("react-plotly.js/factory")
  const fakePlotly = {
    react: async (el: HTMLElement) => { reactCalls.n++; return el },
    newPlot: async (el: HTMLElement) => el,
    purge: () => {},
    Plots: { resize: () => {} },
  }
  const Plot = createPlotlyComponent(fakePlotly as never)
  return { default: () => Plot }
})

import { PlotElement } from "@/components/PlotElement"

class MeasuredResizeObserver {
  constructor(private readonly callback: ResizeObserverCallback) {}
  observe() { this.callback([{ contentRect: { width: 600 } } as ResizeObserverEntry], this as unknown as ResizeObserver) }
  disconnect() {}
}
vi.stubGlobal("ResizeObserver", MeasuredResizeObserver)

const figure = { data: [{ type: "scatter", x: [1], y: [1] }], layout: {} }

it("does not redraw the plot when its host re-renders with the same figure", async () => {
  let rerenderHost = () => {}
  function Host() {
    const [tick, setTick] = useState(0)
    rerenderHost = () => setTick(tick + 1)
    return <div data-tick={tick}><PlotElement figure={figure} /></div>
  }
  render(<Host />)
  await act(async () => {})
  const afterMount = reactCalls.n

  for (let i = 0; i < 20; i++) act(() => rerenderHost())
  await act(async () => {})

  expect(reactCalls.n).toBe(afterMount)
})
