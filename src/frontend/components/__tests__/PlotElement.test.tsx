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
      title: { text: "Host controlled" },
    })

    act(() => resizeObservers[0]!.resize(800))
    expect(plotProps.at(-1)!.layout).toMatchObject({ width: 800, height: 480, autosize: false })
  })
})
