import { act, render } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const plotProps = vi.hoisted(() => [] as Array<{ layout: Record<string, unknown> }>)

vi.mock("next/dynamic", () => ({
  default: () => (props: { layout: Record<string, unknown> }) => {
    plotProps.push(props)
    return null
  },
}))

import { PlotElement } from "@/components/content/PlotElement"

const resizeObservers: TestResizeObserver[] = []

class TestResizeObserver {
  constructor(private readonly callback: ResizeObserverCallback) {
    resizeObservers.push(this)
  }

  observe() {}
  unobserve() {}
  disconnect() {}

  resize(width: number) {
    this.callback([{ contentRect: { width } } as ResizeObserverEntry], this as unknown as ResizeObserver)
  }
}

describe("PlotElement", () => {
  beforeEach(() => {
    plotProps.length = 0
    resizeObservers.length = 0
    vi.stubGlobal("ResizeObserver", TestResizeObserver)
    vi.stubGlobal("getComputedStyle", () => ({
      getPropertyValue: (property: string) => ({
        "--ink": "#1a1816",
        "--surface-elevated": "#f5f0e8",
        "--chart-grid": "#d8d0c4",
        "--font-sans": "Inter",
      })[property] ?? "",
    }))
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it("allocates a chart-height slice for each subplot row", () => {
    render(
      <PlotElement
        figure={{
          data: [],
          layout: {
            yaxis: { domain: [0.55, 1] },
            yaxis2: { domain: [0.55, 1] },
            yaxis3: { domain: [0, 0.45] },
            yaxis4: { domain: [0, 0.45] },
          },
        }}
      />,
    )

    act(() => resizeObservers[0]!.resize(480))
    expect(plotProps.at(-1)!.layout).toMatchObject({ width: 480, height: 600 })
  })

  it("keeps model-provided grid and scene chrome aligned with the active theme", () => {
    render(
      <PlotElement
        figure={{
          data: [],
          layout: {
            xaxis: { gridcolor: "white" },
            xaxis2: { gridcolor: "white" },
            scene: { bgcolor: "black", xaxis: { gridcolor: "white", showbackground: true } },
          },
        }}
      />,
    )

    act(() => resizeObservers[0]!.resize(480))

    // The theme's own colours are not the contract: every axis takes the same theme colour instead of the model's
    type Axis = Record<string, unknown>
    const layout = plotProps.at(-1)!.layout as { xaxis: Axis; xaxis2: Axis; scene: { xaxis: Axis } }
    const grid = layout.xaxis.gridcolor
    expect(grid).not.toBe("white")
    for (const axis of [layout.xaxis, layout.xaxis2, layout.scene.xaxis]) {
      expect(axis).toMatchObject({ gridcolor: grid, zerolinecolor: grid, linecolor: grid })
    }
    expect(layout).toMatchObject({ paper_bgcolor: "transparent", plot_bgcolor: "transparent", scene: { bgcolor: "transparent", xaxis: { showbackground: false } } })
  })
})
