import { act, cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, expect, it, vi } from "vitest"

const markdown = vi.hoisted(() => vi.fn())
vi.mock("@/features/chat/messages/ChatMessage", () => ({
  AssistantMessageContent: ({ content, search_result }: { content: string; search_result?: unknown }) => {
    markdown(content, search_result)
    return <div data-testid="markdown">{content}</div>
  },
  ChatMessage: ({ content }: { content: string }) => <div data-testid="message">{content}</div>,
}))
vi.mock("@/features/chat/messages/ToolCallElement", () => ({ ToolCallElement: () => <div>Full tool output</div> }))

import { ChatPair } from "@/features/chat/ChatPair"

let intersect: IntersectionObserverCallback
const disconnect = vi.fn()
beforeEach(() => {
  markdown.mockClear()
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue({
    top: 2000, bottom: 2096, left: 0, right: 600, width: 600, height: 96, x: 0, y: 2000, toJSON() {},
  })
  disconnect.mockClear()
  vi.stubGlobal("IntersectionObserver", class {
    constructor(callback: IntersectionObserverCallback) { intersect = callback }
    observe() {}
    disconnect = disconnect
  })
})
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

const props = {
  user: { role: "user" as const, content: "Question" },
  assistant: {
    role: "assistant" as const,
    content: "Full answer with $x^2$ and every detail",
    tool_calls: [{ id: "plot", name: "sandbox_plot", arguments: {}, status: "done" as const }],
  },
  globalIndex: 0,
  expanded: false,
  focused: false,
  isStreaming: false,
  onExpand: vi.fn(),
  onCollapse: vi.fn(),
  collapsedRef: vi.fn(),
  onAttachmentClick: vi.fn(),
  onDelete: vi.fn(),
}

it("defers offscreen markdown while retaining the header, tools and accessible answer", () => {
  render(<ChatPair {...props} />)
  expect(markdown).not.toHaveBeenCalled()
  expect(screen.getByRole("button", { name: "Question" })).toBeInTheDocument()
  expect(screen.getByRole("list", { name: "Tools used" })).toHaveTextContent("Interactive plot")
  expect(screen.getByText(props.assistant.content)).toBeInTheDocument()

  act(() => intersect([{ isIntersecting: false } as IntersectionObserverEntry], {} as IntersectionObserver))
  expect(markdown).not.toHaveBeenCalled()
  act(() => intersect([{ isIntersecting: true } as IntersectionObserverEntry], {} as IntersectionObserver))
  expect(screen.getByTestId("markdown")).toHaveTextContent(props.assistant.content)
})

it("loads the complete preview on keyboard focus before intersection", () => {
  render(<ChatPair {...props} />)
  fireEvent.focusIn(screen.getByRole("region", { name: "Answer preview" }))
  expect(screen.getByTestId("markdown")).toHaveTextContent(props.assistant.content)
})

it("gives the collapsed preview the search result so citations resolve to links", () => {
  const search_result = { query: "banach", sources: [], images: [], videos: [], citationMap: {} }
  render(<ChatPair {...props} assistant={{ ...props.assistant, search_result }} />)
  fireEvent.focusIn(screen.getByRole("region", { name: "Answer preview" }))
  expect(markdown).toHaveBeenCalledWith(props.assistant.content, search_result)
})

it("renders full messages and tool output on expansion without waiting for intersection", () => {
  const view = render(<ChatPair {...props} />)
  view.rerender(<ChatPair {...props} expanded />)
  expect(screen.getAllByTestId("message").map((node) => node.textContent)).toEqual([props.user.content, props.assistant.content])
  expect(screen.getByText("Full tool output")).toBeInTheDocument()
  expect(screen.queryByRole("region", { name: "Answer preview" })).not.toBeInTheDocument()
})

it("renders normally when IntersectionObserver is unavailable", () => {
  vi.stubGlobal("IntersectionObserver", undefined)
  render(<ChatPair {...props} />)
  expect(screen.getByTestId("markdown")).toHaveTextContent(props.assistant.content)
})


it("renders already-visible previews before waiting for an observer callback", () => {
  vi.mocked(HTMLElement.prototype.getBoundingClientRect).mockReturnValue({
    top: 100, bottom: 196, left: 0, right: 600, width: 600, height: 96, x: 0, y: 100, toJSON() {},
  })
  render(<ChatPair {...props} />)
  expect(screen.getByTestId("markdown")).toHaveTextContent(props.assistant.content)
})

it("keeps hidden-tab previews deferred until they become visible", () => {
  vi.mocked(HTMLElement.prototype.getBoundingClientRect).mockReturnValue({
    top: 0, bottom: 0, left: 0, right: 0, width: 0, height: 0, x: 0, y: 0, toJSON() {},
  })
  render(<ChatPair {...props} />)
  expect(screen.queryByTestId("markdown")).not.toBeInTheDocument()

  act(() => intersect([{ isIntersecting: true } as IntersectionObserverEntry], {} as IntersectionObserver))
  expect(screen.getByTestId("markdown")).toHaveTextContent(props.assistant.content)
})