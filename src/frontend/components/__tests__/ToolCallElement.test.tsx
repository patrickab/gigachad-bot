import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { ToolCallRecord } from "@/lib/types"

const plotCalls = vi.hoisted(() => vi.fn())

vi.mock("@/components/PlotElement", () => ({
  PlotElement: ({ figure }: { figure: unknown }) => {
    plotCalls(figure)
    return <div data-testid="plot" />
  },
}))

import { ToolCallElement } from "@/components/ToolCallElement"

const sandboxPlotCall: ToolCallRecord = {
  id: "plot-1",
  name: "sandbox_plot",
  arguments: { brief: "Compare the two trends" },
  status: "done",
  detail: {
    brief: "- Compare the two trends\n- Focus on the widening gap",
    figure: { data: [{ type: "scatter" }] },
  },
}

describe("ToolCallElement sandbox plots", () => {
  beforeEach(() => {
    plotCalls.mockReset()
  })

  it("keeps the chart visible while the briefing disclosure toggles", () => {
    render(<ToolCallElement call={sandboxPlotCall} />)

    expect(plotCalls).toHaveBeenCalledWith(sandboxPlotCall.detail!.figure)
    expect(screen.getByTestId("plot")).toBeInTheDocument()
    expect(screen.queryByText("- Compare the two trends")).not.toBeInTheDocument()

    const briefing = screen.getByRole("button", { name: "Briefing" })
    expect(briefing).toHaveAttribute("aria-expanded", "false")
    fireEvent.click(briefing)

    expect(briefing).toHaveAttribute("aria-expanded", "true")
    expect(screen.getByText(/Compare the two trends/)).toBeInTheDocument()
    expect(screen.getByTestId("plot")).toBeInTheDocument()
  })

  it("retains the argument disclosure for other tools", () => {
    render(<ToolCallElement call={{ id: "search-1", name: "web_search", arguments: { query: "plotly docs" }, status: "done" }} />)

    const tool = screen.getByRole("button", { name: /web search/i })
    expect(tool).toHaveAttribute("aria-expanded", "false")
    fireEvent.click(tool)

    expect(tool).toHaveAttribute("aria-expanded", "true")
    expect(screen.getByText("query")).toBeInTheDocument()
  })

  it("surfaces a failed plot's error without a disclosure to open", () => {
    render(<ToolCallElement call={{ id: "plot-2", name: "sandbox_plot", arguments: { brief: "Plot it" }, status: "error", error: "Kernel died" }} />)

    expect(screen.getByText("Kernel died")).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: /interactive plot/i })).not.toBeInTheDocument()
  })

  it("renders an unknown saved tool name as a generic call", () => {
    render(<ToolCallElement call={{ id: "legacy-1", name: "legacy_tool", arguments: { topic: "archived" }, status: "done" }} />)

    const tool = screen.getByRole("button", { name: /legacy_tool/i })
    fireEvent.click(tool)

    expect(screen.getByText("topic")).toBeInTheDocument()
  })
})
