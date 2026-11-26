import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react"
import type { Message } from "@/lib/domain/types"
import { ChatContainer } from "../ChatContainer"

// jsdom has no layout or scrolling, the chat pins itself to the bottom on mount
beforeAll(() => { Element.prototype.scrollIntoView = () => {} })
vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} })

vi.mock("@/features/chat/EnabledToolsContext", () => ({ useEnabledTools: () => ({ enabledTools: [], toggleTool: () => {} }) }))
vi.mock("@/contexts/SettingsContext", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/contexts/SettingsContext")>(),
  useSettings: () => ({ ocrModel: "" }),
  useNeovimSetting: () => ({ available: false, enabled: false, setPreferred: () => {} }),
}))
vi.mock("@/features/chat/messages/ChatMessage", () => ({
  AssistantMessageContent: ({ content }: { content: string }) => <div>{content}</div>,
  ChatMessage: ({ content }: { content: string }) => <div>{content}</div>,
}))
const GREETING = vi.hoisted(() => "What's on your mind?")

// jsdom lays nothing out: size the composer (the element holding the greeting) and the column so the empty state can center
vi.mock("@/hooks/useElementSize", () => ({
  useElementSize: (element: HTMLElement | null) => ({ width: 800, height: !element ? 0 : element.firstElementChild?.textContent === GREETING ? 100 : 600 }),
}))

function renderChat(historyLoading: boolean) {
  render(<ChatContainer chatId="c1" messages={[]} historyLoading={historyLoading} isStreaming={false} onSend={() => {}} onCancel={() => {}} onDeletePair={() => {}} />)
  return { greeting: screen.getByText(GREETING), composerArea: screen.getByText(GREETING).parentElement! }
}

describe("ChatContainer empty state", () => {
  afterEach(cleanup)

  it("centers the composer under the greeting in a chat known to be empty", () => {
    const { greeting, composerArea } = renderChat(false)
    expect(greeting).toHaveAttribute("aria-hidden", "false")
    expect(composerArea.style.transform).not.toBe("translateY(0px)")
  })

  it("keeps the composer at the bottom without a greeting while the history loads", () => {
    const { greeting, composerArea } = renderChat(true)
    expect(greeting).toHaveAttribute("aria-hidden", "true")
    expect(composerArea.style.transform).toBe("translateY(0px)")
  })
})

function pairsOf(count: number, answerTail = ""): Message[] {
  return Array.from({ length: count }, (_, index) => [
    { role: "user" as const, content: `Question ${index}` },
    { role: "assistant" as const, content: `Answer ${index}${index === count - 1 ? answerTail : ""}` },
  ]).flat()
}

describe("ChatContainer column", () => {
  // jsdom has no layout: content height follows the text, scrollTop is stored as set
  const scrollTops = new WeakMap<Element, number>()
  beforeEach(() => {
    vi.spyOn(HTMLElement.prototype, "scrollHeight", "get").mockImplementation(function (this: HTMLElement) { return (this.textContent ?? "").length * 10 })
    vi.spyOn(HTMLElement.prototype, "clientHeight", "get").mockImplementation(() => 100)
    Object.defineProperty(HTMLElement.prototype, "scrollTop", {
      configurable: true,
      get(this: HTMLElement) { return scrollTops.get(this) ?? 0 },
      set(this: HTMLElement, value: number) { scrollTops.set(this, value) },
    })
  })
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    Reflect.deleteProperty(HTMLElement.prototype, "scrollTop")
  })

  type Props = Partial<React.ComponentProps<typeof ChatContainer>> & { messages: Message[] }
  const chat = (props: Props) => <ChatContainer chatId="c1" isStreaming={false} onSend={() => {}} onCancel={() => {}} onDeletePair={() => {}} {...props} />
  const scroller = () => document.querySelector("[data-chat-column]")!.parentElement!
  const isOpen = (question: string) => screen.getByText(question).closest(".gc-turn")?.getAttribute("data-expanded") === "true"

  it("opens at the newest message and follows a streamed answer", () => {
    const { rerender } = render(chat({ messages: pairsOf(3) }))
    expect(scroller().scrollTop).toBe(scroller().scrollHeight)
    expect(isOpen("Question 2")).toBe(true)

    rerender(chat({ messages: pairsOf(3, " and a streamed tail"), isStreaming: true }))
    expect(screen.getByText("Answer 2 and a streamed tail")).toBeTruthy()
    expect(scroller().scrollTop).toBe(scroller().scrollHeight)
  })

  it("mounts the newest page of a long history, earlier pairs on request", () => {
    render(chat({ messages: pairsOf(25) }))
    expect(screen.queryByText("Question 4")).toBeNull()
    expect(screen.getByText("Question 5")).toBeTruthy()
    fireEvent.click(screen.getByRole("button", { name: "Show earlier turns" }))
    expect(screen.getByText("Question 0")).toBeTruthy()
  })

  it("jumps to the revealed question: mounts, opens and scrolls to it", () => {
    const scrolled: Element[] = []
    Element.prototype.scrollIntoView = function (this: Element) { scrolled.push(this) }
    const messages = pairsOf(25)
    const { rerender } = render(chat({ messages, reveal: { qaIndex: null, seq: 0 } }))
    rerender(chat({ messages, reveal: { qaIndex: 1, seq: 1 } }))
    expect(isOpen("Question 1")).toBe(true)
    expect(scrolled.at(-1)?.contains(screen.getByText("Question 1"))).toBe(true)
    Element.prototype.scrollIntoView = () => {}
  })

  it("moves keyboard focus with Ctrl+Up from the composer, Enter collapses the focused pair", () => {
    render(chat({ messages: pairsOf(3) }))
    const composer = screen.getByRole("textbox")
    act(() => { composer.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowUp", ctrlKey: true, bubbles: true, cancelable: true })) })
    expect(screen.getByText("Question 2").closest(".gc-turn")?.hasAttribute("data-focused")).toBe(true)
    act(() => { document.body.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true, cancelable: true })) })
    expect(isOpen("Question 2")).toBe(false)
  })

  it("shows the branch separator after the pair the chat branched at", () => {
    render(chat({ messages: pairsOf(3), branchMessageIdx: 1 }))
    const separator = screen.getByRole("separator")
    expect(separator).toHaveTextContent("branched@1")
    expect(separator.previousElementSibling?.textContent).toContain("Question 1")
  })

  it("keeps a pair's open state when an earlier pair is deleted", () => {
    const messages = pairsOf(3)
    const { rerender } = render(chat({ messages }))
    fireEvent.click(screen.getByRole("button", { name: "Question 1" }))
    expect(isOpen("Question 1")).toBe(true)

    rerender(chat({ messages: messages.slice(2) }))
    expect(isOpen("Question 1")).toBe(true)
  })
})
