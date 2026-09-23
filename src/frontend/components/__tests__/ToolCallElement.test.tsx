import { fireEvent, render, screen } from "@testing-library/react"
import { ApiError } from "@/lib/api"
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

const markdownCalls = vi.hoisted(() => vi.fn())

vi.mock("@/components/LaTeXMarkdown", () => ({
  LaTeXMarkdown: ({ content }: { content: string }) => {
    markdownCalls(content)
    return <div data-testid="mermaid-diagram">{content}</div>
  },
}))

const saveNotebookCalls = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api", () => ({
  ApiError: class ApiError extends Error {
    constructor(message: string, readonly status: number) {
      super(message)
    }
  },
  saveNotebook: (...args: unknown[]) => saveNotebookCalls(...args),
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
    markdownCalls.mockReset()
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


  it("renders Mermaid source directly in a diagram tool call", () => {
    render(<ToolCallElement call={{
      id: "diagram-1",
      name: "diagram",
      arguments: {},
      status: "done",
      detail: { mermaid: "flowchart LR\nA --> B" },
    }} />)

    expect(markdownCalls).toHaveBeenCalledWith("```mermaid\nflowchart LR\nA --> B\n```")
    expect(screen.getByTestId("mermaid-diagram")).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: /^diagram$/i })).not.toBeInTheDocument()
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

const notebookEditCall: ToolCallRecord = {
  id: "nb-1",
  name: "notebook_edit",
  arguments: { prompt: "swap the second cell and add one" },
  status: "done",
  detail: {
    before: "# %%\na = 1\n\n# %%\nb = 2\n",
    after: "# %%\na = 1\n\n# %%\nx = 9\n\n# %%\nz = 0\n",
    outputs: { preserved: [{ type: "stdout", text: "old result" }] },
    revision_id: "rev-7",
  },
}

describe("ToolCallElement notebook edits", () => {
  beforeEach(() => {
    codeBlockCalls.mockReset()
    saveNotebookCalls.mockReset()
  })

  it("renders the notebook card collapsed with cell counts, not the diff", () => {
    render(<ToolCallElement call={notebookEditCall} chatId="chat-1" />)

    expect(screen.getByText("Notebook updated · +1 ~1 −0")).toBeInTheDocument()
    expect(screen.queryByTestId("code-block")).not.toBeInTheDocument()
    expect(codeBlockCalls).not.toHaveBeenCalled()
  })

  it("expands into the highlighted line diff with + and − lines", () => {
    render(<ToolCallElement call={notebookEditCall} chatId="chat-1" />)

    const expander = screen.getByRole("button", { name: /Notebook updated/ })
    expect(expander).toHaveAttribute("aria-expanded", "false")
    fireEvent.click(expander)

    expect(expander).toHaveAttribute("aria-expanded", "true")
    expect(codeBlockCalls).toHaveBeenCalledTimes(1)
    const [codeString, language] = codeBlockCalls.mock.calls[0]
    expect(language).toBe("diff")
    expect(codeString).toContain("- b = 2")
    expect(codeString).toContain("+ x = 9")
    expect(codeString).toContain("+ z = 0")
    expect(screen.getByText("Undo")).toBeInTheDocument()
  })


  it("undo PUTs the pre-edit source at the tool call's revision", async () => {
    saveNotebookCalls.mockResolvedValue({ revision_id: "rev-8" })
    render(<ToolCallElement call={notebookEditCall} chatId="chat-1" />)

    fireEvent.click(screen.getByRole("button", { name: /Notebook updated/ }))
    fireEvent.click(screen.getByRole("button", { name: "Undo" }))

    expect(saveNotebookCalls).toHaveBeenCalledWith(
      "chat-1",
      "# %%\na = 1\n\n# %%\nb = 2\n",
      { preserved: [{ type: "stdout", text: "old result" }] },
      "rev-7",
    )
  })

  it("disables undo after a 409 stale-revision rejection", async () => {
    saveNotebookCalls.mockRejectedValue(new ApiError("Stale notebook revision", 409))
    render(<ToolCallElement call={notebookEditCall} chatId="chat-1" />)

    fireEvent.click(screen.getByRole("button", { name: /Notebook updated/ }))
    fireEvent.click(screen.getByRole("button", { name: "Undo" }))

    const stale = await screen.findByRole("button", { name: /stale revision/ })
    expect(stale).toBeDisabled()
    expect(screen.getByText(/notebook changed elsewhere/i)).toBeInTheDocument()
  })

})
