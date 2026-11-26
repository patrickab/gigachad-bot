import { act, fireEvent, render, screen, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

vi.mock("@/components/content/LaTeXMarkdown", () => ({
  LaTeXMarkdown: ({ content }: { content: string }) => <div data-testid="markdown">{content}</div>,
}))

import { TabContext, type Tab } from "@/contexts/TabContext"
import "../FactoryNotices"
import "../FactoryTool"
import { FACTORY_VIEW_STORAGE_KEY } from "../factoryView"
import { PiTranscript } from "../PiTranscript"
import { renderCustomMessage, renderToolCall } from "../renderers"
import { EMPTY_TRANSCRIPT, type Transcript } from "../transcript"
import type { AgentMessage, CustomMessage, ToolCall, ToolRun } from "../types"
import { CARD_FIXTURES } from "./factoryCardFixtures"

const NOTICE_TEXT = "factory: task c8eb1223 asks\nShould quiet cards start collapsed?"
const notice = (details?: unknown): CustomMessage => ({ role: "custom", customType: "factory-notice", content: NOTICE_TEXT, display: true, details, timestamp: 0 })
const showNotice = (details?: unknown) => render(<>{renderCustomMessage(notice(details), () => <p>raw text</p>)}</>)

const RESULT_TEXT = "message sent to task c8eb1223\nKeep quiet cards collapsed."
const factoryCall = { type: "toolCall", id: "c1", name: "factory", arguments: { action: "tasks.message", id: "c8eb1223", text: "Keep quiet cards collapsed." } } as ToolCall
const run = (details?: unknown): ToolRun => ({ result: { content: [{ type: "text", text: RESULT_TEXT }], details }, isError: false, running: false })
const showTool = (details?: unknown) => render(<>{renderToolCall(factoryCall, run(details), () => <p>default row</p>)}</>)

beforeEach(() => sessionStorage.clear())
afterEach(() => vi.restoreAllMocks())

describe("a factory notice", () => {
  it("with details.card shows the card, not the agent's text", () => {
    const { container } = showNotice({ key: "task:c8eb1223:question:1", card: CARD_FIXTURES.task })
    expect(screen.getByText(CARD_FIXTURES.task.title)).toBeTruthy()
    expect(container.querySelector("[data-factory-notice='task:c8eb1223:question:1']")).not.toBeNull()
    expect(screen.queryByText(/Should quiet cards/)).toBeNull()
  })

  it("without details.card shows its text as before", () => {
    showNotice({ key: "k" })
    expect(screen.getByRole("button", { name: "task c8eb1223 asks" })).toBeTruthy()
    expect(screen.queryByRole("button", { name: "Show the agent's text" })).toBeNull()
  })

  it("switches per card to exactly the message's content and back", () => {
    const { container } = showNotice({ key: "k", card: CARD_FIXTURES.task })
    fireEvent.click(screen.getByRole("button", { name: "Show the agent's text" }))
    expect(container.querySelector("[data-agent-text]")?.textContent).toBe(NOTICE_TEXT)
    fireEvent.click(screen.getByRole("button", { name: "Show the card" }))
    expect(container.querySelector("[data-agent-text]")).toBeNull()
  })
})

describe("a factory tool result", () => {
  it("with details.card shows the card", () => {
    showTool({ result: { ok: true }, card: CARD_FIXTURES.decision })
    expect(screen.getByText(CARD_FIXTURES.decision.title)).toBeTruthy()
    expect(screen.queryByText("message · c8eb1223")).toBeNull()
  })

  it("without details.card keeps the factory row", () => {
    showTool({ result: { ok: true } })
    expect(screen.getByText("message · c8eb1223")).toBeTruthy()
  })

  it("an error result keeps the factory row", () => {
    render(<>{renderToolCall(factoryCall, { ...run({ card: CARD_FIXTURES.decision }), isError: true }, () => <p>default row</p>)}</>)
    expect(screen.getByText("message · c8eb1223")).toBeTruthy()
  })

  it("switches to exactly the result's text", () => {
    const { container } = showTool({ result: {}, card: CARD_FIXTURES.decision })
    fireEvent.click(screen.getByRole("button", { name: "Show the agent's text" }))
    expect(container.querySelector("[data-agent-text]")?.textContent).toBe(RESULT_TEXT)
  })
})

const TAB = { id: "tab-1" } as Tab
const fileLinks = { base: "/repo", open: () => {} }

function inTab(transcript: Transcript) {
  return (
    <TabContext.Provider value={{ tab: TAB, isActive: true, setModeLabel: vi.fn(), updateConfig: vi.fn(), setAppMode: vi.fn(), setWorkspaceItem: vi.fn(), setHistoryFile: vi.fn(), setTitle: vi.fn() }}>
      <PiTranscript transcript={transcript} fileLinks={fileLinks} />
    </TabContext.Provider>
  )
}

// Turn 1 holds the coder's question, turn 2 the supervisor's answer to it, live so its tool row stays unfolded
const messages: AgentMessage[] = [
  { role: "user", content: "Start the wave", timestamp: 1 },
  notice({ key: "task:c8eb1223:question:1", card: CARD_FIXTURES.task }),
  { role: "user", content: "Answer it", timestamp: 3 },
  { role: "assistant", content: [factoryCall], provider: "anthropic", model: "claude", stopReason: "toolUse", timestamp: 4 },
  { role: "toolResult", toolCallId: "c1", toolName: "factory", content: [{ type: "text", text: RESULT_TEXT }], details: { result: {}, card: CARD_FIXTURES.decision }, isError: false, timestamp: 5 },
]
const transcript: Transcript = { ...EMPTY_TRANSCRIPT, messages, entryIds: messages.map((_, index) => `e${index}`), running: true }

describe("factory events in a Pi tab", () => {
  it("a decision scrolls to the notice it answers", () => {
    const scrolled: Element[] = []
    Element.prototype.scrollIntoView = function (this: Element) { scrolled.push(this) }
    const { container } = render(inTab(transcript))
    act(() => fireEvent.click(screen.getByRole("button", { name: "Show what this answers" })))
    const target = container.querySelector("[data-factory-notice='task:c8eb1223:question:1']")
    expect(target).not.toBeNull()
    expect(scrolled).toContain(target)
  })

  it("Alt+V switches every card of the tab, remembered per tab", () => {
    const { container, unmount } = render(inTab(transcript))
    expect(container.querySelectorAll("[data-agent-text]")).toHaveLength(0)
    act(() => { fireEvent.keyDown(window, { key: "v", code: "KeyV", altKey: true }) })
    const texts = [...container.querySelectorAll("[data-agent-text]")].map((element) => element.textContent)
    expect(texts).toEqual([NOTICE_TEXT, RESULT_TEXT])
    expect(sessionStorage.getItem(`${FACTORY_VIEW_STORAGE_KEY}:tab-1`)).toBe("text")
    unmount()
    const again = render(inTab(transcript))
    expect(again.container.querySelectorAll("[data-agent-text]")).toHaveLength(2)
    // A card's own switch flips just that card
    fireEvent.click(within(again.container.querySelector("[data-factory-notice]")!).getByRole("button", { name: "Show the card" }))
    expect(again.container.querySelectorAll("[data-agent-text]")).toHaveLength(1)
  })

  it("Alt+V leaves a terminal's keys alone", () => {
    const { container } = render(<>{inTab(transcript)}<div data-terminal tabIndex={0} data-testid="term" /></>)
    act(() => { fireEvent.keyDown(screen.getByTestId("term"), { key: "v", code: "KeyV", altKey: true }) })
    expect(container.querySelectorAll("[data-agent-text]")).toHaveLength(0)
  })
})
