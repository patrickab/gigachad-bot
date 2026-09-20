import { act, render } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const plotProps = vi.hoisted(() => [] as Array<{ layout: Record<string, unknown> }>)

vi.mock("next/dynamic", () => ({
  default: () => (props: { layout: Record<string, unknown> }) => {
    plotProps.push(props)
    return null
  },
}))

import { PlotElement } from "@/components/PlotElement"

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
        "--divider": "#d8d0c4",
        "--font-sans": "Inter",
      })[property] ?? "",
    }))
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it("uses the measured host width and recomputes bounded height on resize", () => {
    render(
      <PlotElement
        figure={{
          data: [],
          layout: { width: 1400, height: 80, autosize: true, title: { text: "Host controlled" } },
        }}
      />,
    )

    expect(resizeObservers).toHaveLength(1)

    act(() => resizeObservers[0]!.resize(480))
    expect(plotProps).toHaveLength(1)
    expect(plotProps.at(-1)!.layout).toMatchObject({
      width: 480,
      height: 300,
      autosize: false,
      hovermode: false,
      scene: { aspectmode: "cube" },
      title: { text: "Host controlled" },
    })

    act(() => resizeObservers[0]!.resize(800))
    expect(plotProps.at(-1)!.layout).toMatchObject({ width: 800, height: 480, autosize: false })
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
            scene: { bgcolor: "black", xaxis: { gridcolor: "white" } },
          },
        }}
      />,
    )

    act(() => resizeObservers[0]!.resize(480))

    expect(plotProps.at(-1)!.layout).toMatchObject({
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      font: { color: "#1a1816", family: "Inter" },
      xaxis: { gridcolor: "#d8d0c4", zerolinecolor: "#d8d0c4", linecolor: "#d8d0c4" },
      xaxis2: { gridcolor: "#d8d0c4", zerolinecolor: "#d8d0c4", linecolor: "#d8d0c4" },
      scene: {
        bgcolor: "transparent",
        xaxis: { gridcolor: "#d8d0c4", zerolinecolor: "#d8d0c4", linecolor: "#d8d0c4" },
      },
    })
  })
})
