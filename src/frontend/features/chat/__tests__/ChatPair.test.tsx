import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, expect, it, vi } from "vitest"

const markdown = vi.hoisted(() => vi.fn())
vi.mock("@/features/chat/messages/ChatMessage", () => ({
  AssistantMessageContent: ({ content, search_result }: { content: string; search_result?: unknown }) => {
    markdown(content, search_result)
    return <div data-testid="markdown">{content}</div>
  },
  ChatMessage: ({ content }: { content: string }) => <div data-testid="message">{content}</div>,
}))
vi.mock("@/features/chat/toolRenderers", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/features/chat/toolRenderers")>(),
  ChatToolCall: () => <div>Full tool output</div>,
}))

import { ChatPairBody, ChatPairPreview, chatPairPreview, pairLabel } from "@/features/chat/ChatPair"

beforeEach(() => markdown.mockClear())
afterEach(cleanup)

const assistant = {
  role: "assistant" as const,
  content: "Full answer with $x^2$ and every detail",
  tool_calls: [{ id: "plot", name: "sandbox_plot", arguments: {}, status: "done" as const }],
}

it("previews the tools used and the answer", () => {
  render(<ChatPairPreview assistant={assistant} />)
  expect(screen.getByRole("list", { name: "Tools used" })).toHaveTextContent("Interactive plot")
  expect(screen.getByTestId("markdown")).toHaveTextContent(assistant.content)
})

it("renders only the head of a long answer until the preview is scrolled", () => {
  const content = Array.from({ length: 40 }, (_, index) => `Paragraph ${index} ${"word ".repeat(30)}`).join("\n\n")
  render(<ChatPairPreview assistant={{ ...assistant, content }} />)
  const shown = screen.getByTestId("markdown").textContent ?? ""
  expect(content.startsWith(shown.trimEnd())).toBe(true)
  expect(shown.length).toBeLessThan(content.length / 2)

  fireEvent.scroll(screen.getByRole("region", { name: "Answer preview" }))
  expect(screen.getByTestId("markdown").textContent).toBe(content)
})

it("gives the collapsed preview the search result so citations resolve to links", () => {
  const search_result = { query: "banach", sources: [], images: [], videos: [], citationMap: {} }
  render(<ChatPairPreview assistant={{ ...assistant, search_result }} />)
  expect(markdown).toHaveBeenCalledWith(assistant.content, search_result)
})

it("has no preview for an empty answer without tools, so the column reserves no space for it", () => {
  expect(chatPairPreview({ role: "assistant", content: "" })).toBeNull()
  expect(chatPairPreview({ ...assistant, content: "" })).not.toBeNull()
  expect(chatPairPreview({ role: "assistant", content: "Answer" })).not.toBeNull()
})

it("renders the full messages and tool output of an open pair", () => {
  render(<ChatPairBody user={{ role: "user", content: "Question" }} assistant={assistant} globalIndex={0} qaIndex={0} isStreaming={false} />)
  expect(screen.getAllByTestId("message").map((node) => node.textContent)).toEqual(["Question", assistant.content])
  expect(screen.getByText("Full tool output")).toBeInTheDocument()
  expect(screen.queryByRole("region", { name: "Answer preview" })).not.toBeInTheDocument()
})

it("labels a question by its first line, else its attachments", () => {
  expect(pairLabel({ role: "user", content: "First line\nsecond" })).toBe("First line")
  expect(pairLabel({ role: "user", content: "", attachments: [{ name: "a.pdf" }, { name: "b.png" }] as never })).toBe("a.pdf, b.png")
})
