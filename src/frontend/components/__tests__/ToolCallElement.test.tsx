import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { ToolCallRecord } from "@/lib/types"

const plotCalls = vi.hoisted(() => vi.fn())
const codeBlockCalls = vi.hoisted(() => vi.fn())

vi.mock("@/components/PlotElement", () => ({
  PlotElement: ({ figure }: { figure: unknown }) => {
    plotCalls(figure)
    return <div data-testid="plot" />
  },
}))

vi.mock("@/components/CodeBlock", () => ({
  CodeBlock: ({ codeString, language }: { codeString: string; language: string }) => {
    codeBlockCalls(codeString, language)
    return <div data-testid="code-block">{codeString}</div>
  },
}))

import { ToolCallElement } from "@/components/ToolCallElement"

const sandboxPlotCall: ToolCallRecord = {
  id: "plot-1",
  name: "sandbox_plot",
  arguments: { code: "print(fig.to_json())" },
  status: "done",
  detail: {
    brief: "- Compare the two trends\n- Focus on the widening gap",
    figure: { data: [{ type: "scatter" }] },
    script: "fig = go.Figure()\nprint(fig.to_json())",
  },
}

describe("ToolCallElement sandbox plots", () => {
  beforeEach(() => {
    plotCalls.mockReset()
    codeBlockCalls.mockReset()
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

  it("keeps the script hidden until the code disclosure toggles", () => {
    render(<ToolCallElement call={sandboxPlotCall} />)

    expect(codeBlockCalls).not.toHaveBeenCalled()
    expect(screen.queryByTestId("code-block")).not.toBeInTheDocument()

    const code = screen.getByRole("button", { name: "Code" })
    expect(code).toHaveAttribute("aria-expanded", "false")
    fireEvent.click(code)

    expect(code).toHaveAttribute("aria-expanded", "true")
    expect(codeBlockCalls).toHaveBeenCalledWith(sandboxPlotCall.detail!.script, "python")
    expect(screen.getByTestId("code-block")).toBeInTheDocument()
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

  it("shows each stage status and its decimal duration", () => {
    render(<ToolCallElement call={{
      id: "search-stages",
      name: "web_search",
      arguments: { query: "stages" },
      status: "done",
      stages: [
        { id: "stage-1", label: "Planning search", status: "done", started_at: 1, duration: 0.2 },
        { id: "stage-2", label: "Searching sources", status: "done", started_at: 1.2, duration: 1.4 },
      ],
    }} />)

    expect(screen.getByText("0.2s").parentElement).toHaveTextContent(/0\.2s\s*Planning search/)
    expect(screen.getByText("1.4s").parentElement).toHaveTextContent(/1\.4s\s*Searching sources/)
  })
})
