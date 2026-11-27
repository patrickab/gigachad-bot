import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { ToolCallRecord } from "@/lib/domain/types"

vi.mock("@/components/content/PlotElement", () => ({
  PlotElement: () => <div data-testid="plot" />,
}))

vi.mock("@gigachad/chat-ui", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@gigachad/chat-ui")>()),
  CodeBlock: ({ codeString }: { codeString: string }) => <div data-testid="code-block">{codeString}</div>,
}))

vi.mock("@/components/content/LaTeXMarkdown", () => ({
  LaTeXMarkdown: ({ content }: { content: string }) => <div data-testid="mermaid-diagram">{content}</div>,
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

import { ChatToolCall } from "@/features/chat/toolRenderers"

describe("chat tool rows: sandbox plots", () => {
  it("surfaces a failed plot's error without opening its row", () => {
    render(<ChatToolCall record={{ id: "plot-2", name: "sandbox_plot", arguments: { brief: "Plot it" }, status: "error", error: "Kernel died" }} />)

    expect(screen.getByRole("button", { name: /interactive plot/i })).toHaveAttribute("aria-expanded", "false")
    expect(screen.getByText("Kernel died")).toBeInTheDocument()
  })

  it("renders an unknown saved tool name as a generic call", () => {
    render(<ChatToolCall record={{ id: "legacy-1", name: "legacy_tool", arguments: { topic: "archived" }, status: "done" }} />)

    const tool = screen.getByRole("button", { name: /legacy_tool/i })
    fireEvent.click(tool)
    fireEvent.click(screen.getByRole("button", { name: "Show full input" }))

    expect(screen.getByText("topic")).toBeInTheDocument()
  })
})

describe("chat tool rows: learn sessions", () => {
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
    render(<ChatToolCall record={learnCall} />)

    expect(screen.getByText("Learn session")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Understand Fourier series" }))

    expect(learn.openItem).toHaveBeenCalledWith({
      kind: "learn", path: "", projectSlug: "physics", learn: { kind: "session", sessionId: "chat_mvtsqwg2_5e0e6463" },
    }, true)
    // A session the sidebar does not list yet is looked up at once.
    expect(learn.sessions.lookUp).toHaveBeenCalledWith("chat_mvtsqwg2_5e0e6463")
  })

  it("names a session LearnHarness no longer has instead of linking it", () => {
    learn.sessions.missing = new Set(["chat_mvtsqwg2_5e0e6463"])
    render(<ChatToolCall record={learnCall} />)

    expect(screen.queryByRole("button", { name: "Understand Fourier series" })).not.toBeInTheDocument()
    expect(screen.getByText("Understand Fourier series (deleted)")).toBeInTheDocument()
  })
})
