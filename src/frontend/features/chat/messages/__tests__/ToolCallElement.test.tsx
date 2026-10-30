import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { ToolCallRecord } from "@/lib/domain/types"

const plotCalls = vi.hoisted(() => vi.fn())
const codeBlockCalls = vi.hoisted(() => vi.fn())

vi.mock("@/components/content/PlotElement", () => ({
  PlotElement: ({ figure }: { figure: unknown }) => {
    plotCalls(figure)
    return <div data-testid="plot" />
  },
}))

vi.mock("@gigachad/chat-ui", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@gigachad/chat-ui")>()),
  CodeBlock: ({ codeString, language }: { codeString: string; language: string }) => {
    codeBlockCalls(codeString, language)
    return <div data-testid="code-block">{codeString}</div>
  },
}))

const markdownCalls = vi.hoisted(() => vi.fn())

vi.mock("@/components/content/LaTeXMarkdown", () => ({
  LaTeXMarkdown: ({ content }: { content: string }) => {
    markdownCalls(content)
    return <div data-testid="mermaid-diagram">{content}</div>
  },
}))

const learn = vi.hoisted(() => ({
  learnUrl: "http://127.0.0.1:30181",
  openItem: vi.fn(),
  sessions: {
    sessions: [] as import("@/lib/domain/types").LearnSession[], error: null, refresh: vi.fn(), claim: vi.fn(),
    lookUp: vi.fn<(sessionId: string) => Promise<void>>(async () => {}), missing: new Set<string>(),
  },
}))
vi.mock("@/features/learn/LearnProvider", () => ({ useLearn: () => learn }))

import { ToolCallElement } from "@/features/chat/messages/ToolCallElement"

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

describe("ToolCallElement learn sessions", () => {
  const learnCall: ToolCallRecord = {
    id: "learn-1",
    name: "start_learn_session",
    arguments: { goal: "I want to understand Fourier series properly." },
    status: "done",
    summary: "Session started",
    detail: { session_id: "chat_mvtsqwg2_5e0e6463", title: "Understand Fourier series", project_slug: "physics" },
  }

  beforeEach(() => {
    learn.sessions.sessions = []
    learn.sessions.missing = new Set()
    learn.sessions.lookUp.mockClear()
    learn.openItem.mockReset()
  })

  it("links the started session, opening the learn view at it in a new tab of its project", () => {
    render(<ToolCallElement call={learnCall} />)

    expect(screen.getByText("Learn session")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Understand Fourier series" }))

    expect(learn.openItem).toHaveBeenCalledWith({
      kind: "learn", path: "", projectSlug: "physics", learn: { kind: "session", sessionId: "chat_mvtsqwg2_5e0e6463" },
    }, true)
    // A session the sidebar does not list yet is looked up at once.
    expect(learn.sessions.lookUp).toHaveBeenCalledWith("chat_mvtsqwg2_5e0e6463")
  })

  it("looks up no session the sidebar already lists", () => {
    learn.sessions.sessions = [{ id: "chat_mvtsqwg2_5e0e6463", title: "Understand Fourier series", kind: null, busy: true, project_slug: "physics" }]
    render(<ToolCallElement call={learnCall} />)

    expect(learn.sessions.lookUp).not.toHaveBeenCalled()
  })

  it("names a session LearnHarness no longer has instead of linking it", () => {
    learn.sessions.missing = new Set(["chat_mvtsqwg2_5e0e6463"])
    render(<ToolCallElement call={learnCall} />)

    expect(screen.queryByRole("button", { name: "Understand Fourier series" })).not.toBeInTheDocument()
    expect(screen.getByText("Understand Fourier series (deleted)")).toBeInTheDocument()
  })

  it("shows no link while the session is being created", () => {
    render(<ToolCallElement call={{ ...learnCall, status: "running", summary: undefined, detail: undefined }} />)

    expect(screen.queryByRole("button", { name: "Understand Fourier series" })).not.toBeInTheDocument()
  })
})
