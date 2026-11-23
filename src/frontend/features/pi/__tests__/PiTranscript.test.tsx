import { act, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { PiTranscript, TURN_PAGE_SIZE } from "../PiTranscript"
import { clearPiSearchHit, focusPiSearchHit, usePiSearchFocus } from "../searchFocus"
import { EMPTY_TRANSCRIPT, transcriptReducer, type SessionEntry, type Transcript } from "../transcript"
import type { AgentMessage, AssistantMessage } from "../types"

// The hits of the whole session are read through the API: a failure keeps the listed ones
vi.mock("@/lib/backend/api", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/lib/backend/api")>(),
  fetchPiSessionMatches: () => Promise.reject(new Error("offline")),
}))

const fileLinks = { base: "/repo", open: () => {} }

function history(turnCount: number): AgentMessage[] {
  return Array.from({ length: turnCount }, (_, index): AgentMessage[] => [
    { role: "user", content: `Prompt ${index}`, timestamp: index * 10 },
    { role: "assistant", content: [{ type: "text", text: `Answer ${index}` }], provider: "anthropic", model: "claude", stopReason: "stop", timestamp: index * 10 + 1 },
  ]).flat()
}

function transcriptOf(messages: AgentMessage[], extra: Partial<Transcript> = {}): Transcript {
  return { ...EMPTY_TRANSCRIPT, messages, entryIds: messages.map((_, index) => `e${index}`), ...extra }
}

const streamingReply = (text: string): AssistantMessage => ({ role: "assistant", content: [{ type: "text", text }], provider: "anthropic", model: "claude", stopReason: "pending", timestamp: 99_999 })

// jsdom has no layout: content height follows the text, the viewport is 100px, scrollTop is stored as set
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
  vi.restoreAllMocks()
  Reflect.deleteProperty(HTMLElement.prototype, "scrollTop")
})

describe("PiTranscript window", () => {
  it("mounts the newest page of turns only and follows the end while streaming", () => {
    const messages = history(45)
    const { container, rerender } = render(<PiTranscript transcript={transcriptOf(messages, { running: true })} fileLinks={fileLinks} />)
    const scroller = container.firstElementChild as HTMLElement
    const firstMounted = 45 - TURN_PAGE_SIZE
    expect(screen.queryByText(`Prompt ${firstMounted - 1}`)).toBeNull()
    expect(screen.getByText(`Prompt ${firstMounted}`)).toBeTruthy()
    expect(scroller.scrollTop).toBe(scroller.scrollHeight)

    rerender(<PiTranscript transcript={transcriptOf(messages, { running: true, streaming: streamingReply("Streaming a long answer") })} fileLinks={fileLinks} />)
    const grown = scroller.scrollHeight
    expect(scroller.textContent).toContain("Streaming a long answer")
    expect(scroller.scrollTop).toBe(grown)
  })

  it("keeps the read window while the reader is away from the end, and loads earlier turns on request", () => {
    const messages = history(45)
    const { container, rerender } = render(<PiTranscript transcript={transcriptOf(messages)} fileLinks={fileLinks} />)
    const scroller = container.firstElementChild as HTMLElement
    scroller.scrollTop = 0
    fireEvent.scroll(scroller)

    rerender(<PiTranscript transcript={transcriptOf(history(60))} fileLinks={fileLinks} />)
    expect(screen.getByText(`Prompt ${45 - TURN_PAGE_SIZE}`)).toBeTruthy()
    expect(scroller.scrollTop).toBe(0)

    fireEvent.click(screen.getByRole("button", { name: "Show earlier turns" }))
    expect(screen.getByText(`Prompt ${45 - 2 * TURN_PAGE_SIZE}`)).toBeTruthy()
  })

  it("offers branching from a settled turn's last persisted entry", () => {
    const onBranch = vi.fn()
    render(<PiTranscript transcript={transcriptOf(history(2))} fileLinks={fileLinks} onBranch={onBranch} />)
    fireEvent.click(screen.getByRole("button", { name: "Branch session from this point" }))
    expect(onBranch).toHaveBeenCalledWith("e3")
  })

  it("edits a turn by branching before its prompt with the prompt", () => {
    const onBranch = vi.fn()
    render(<PiTranscript transcript={transcriptOf(history(2))} fileLinks={fileLinks} onBranch={onBranch} />)
    fireEvent.click(screen.getByRole("button", { name: "Edit in a new branch" }))
    expect(onBranch).toHaveBeenCalledWith("e1", "Prompt 1")
  })

  it("previews an older turn's answer under its question with activity folded, as when opened", () => {
    const messages: AgentMessage[] = [
      { role: "user", content: "Run it", timestamp: 1 },
      { role: "assistant", content: [{ type: "toolCall", id: "t1", name: "bash", arguments: { command: "ls" } }], provider: "anthropic", model: "claude", stopReason: "toolUse", timestamp: 2 },
      { role: "assistant", content: [{ type: "text", text: "Done" }], provider: "anthropic", model: "claude", stopReason: "stop", timestamp: 3 },
      ...history(1).map((message) => ({ ...message, timestamp: message.timestamp + 10 })),
    ]
    render(<PiTranscript transcript={transcriptOf(messages)} fileLinks={fileLinks} />)
    expect(screen.getByText("Done")).toBeTruthy()
    expect(screen.getByRole("button", { name: "Process details · 1 message · 1 tool call" })).toBeTruthy()
    fireEvent.click(screen.getByText("Run it"))
    expect(screen.getByRole("button", { name: "Process details · 1 message · 1 tool call" })).toBeTruthy()
    expect(screen.getAllByRole("button", { name: "Collapse question and answer" }).length).toBeGreaterThan(2)
  })
})

describe("PiTranscript entry edits", () => {
  const entry = (id: string, parentId: string | null, fields: Record<string, unknown>): SessionEntry => ({ id, parentId, timestamp: "2026-11-19T19:02:30.251Z", type: "message", ...fields })
  const assistant = (fields: Partial<AssistantMessage>): AssistantMessage => ({ role: "assistant", content: [], provider: "anthropic", model: "claude-haiku-4-5", stopReason: "stop", timestamp: 2, ...fields })
  const snapshotOf = (entries: SessionEntry[]): Transcript =>
    transcriptReducer(EMPTY_TRANSCRIPT, { kind: "snapshot", entries: { entries, leafId: entries.at(-1)!.id }, running: false, queue: { steering: [], followUp: [] } })

  it("shows an attempt Pi's auto-retry omitted as one quiet line before the retry's answer", () => {
    // Shaped like a real session: the failed attempt, Pi's `context_edit` omitting it, then the retry
    const transcript = snapshotOf([
      entry("u1", null, { message: { role: "user", content: "Reply with just: ok", timestamp: 1 } }),
      entry("9d1d64ed", "u1", { message: assistant({ stopReason: "error", errorMessage: "Request timed out.", timestamp: 2 }) }),
      entry("edit1", "9d1d64ed", { type: "context_edit", targetId: "9d1d64ed", replacement: null }),
      entry("a2", "edit1", { message: assistant({ content: [{ type: "text", text: "ok" }], timestamp: 3 }) }),
    ])
    render(<PiTranscript transcript={transcript} fileLinks={fileLinks} />)
    expect(screen.getByText("Retried after: Request timed out")).toBeTruthy()
    expect(screen.getByText("ok")).toBeTruthy()
    expect(screen.queryByText("Request timed out.")).toBeNull()
    expect(screen.queryByText(/Omitted from context/)).toBeNull()
  })

  it("keeps the omitted marker on an omitted answer that did not fail", () => {
    const transcript = snapshotOf([
      entry("u1", null, { message: { role: "user", content: "Hi", timestamp: 1 } }),
      entry("a1", "u1", { message: assistant({ content: [{ type: "text", text: "Hello" }] }) }),
      entry("edit1", "a1", { type: "context_edit", targetId: "a1", replacement: null }),
    ])
    render(<PiTranscript transcript={transcript} fileLinks={fileLinks} />)
    expect(screen.getByText("Omitted from context")).toBeTruthy()
    expect(screen.getByText("Hello")).toBeTruthy()
    expect(screen.queryByText(/Retried after/)).toBeNull()
  })
})

describe("PiTranscript search hit", () => {
  const hit = (entryId: string, ordinal = 0) => ({ entryId, ordinal, quality: 1, before: "", match: "needle", after: "" })
  /** Highlight registry and the calls jsdom lacks. */
  const registry = new Map<string, { ranges: Range[] }>()
  let scrolled: Element[]

  beforeEach(() => {
    registry.clear()
    scrolled = []
    class FakeHighlight { ranges: Range[]; constructor(...ranges: Range[]) { this.ranges = ranges } add(range: Range) { this.ranges.push(range) } }
    vi.stubGlobal("Highlight", FakeHighlight)
    Object.defineProperty(CSS, "highlights", { configurable: true, value: registry })
    Object.defineProperty(CSS, "escape", { configurable: true, value: (value: string) => value })
    Element.prototype.scrollIntoView = function (this: Element) { scrolled.push(this) }
    Element.prototype.scrollTo = () => {}
    Range.prototype.getBoundingClientRect = () => new DOMRect(0, 40, 10, 10)
  })
  afterEach(() => {
    clearPiSearchHit()
    vi.unstubAllGlobals()
    Reflect.deleteProperty(CSS, "highlights")
  })

  function Harness({ transcript }: { transcript: Transcript }) {
    return <PiTranscript transcript={transcript} fileLinks={fileLinks} search={usePiSearchFocus("s1")} />
  }

  /** The text each painted range covers. */
  const painted = (name: string) => registry.get(name)?.ranges.map((range) => range.toString())

  it("widens the window and opens the turn, scrolls to the hit and paints every hit with the current one apart", () => {
    const messages = history(45)
    messages[0] = { role: "user", content: "an old needle here", timestamp: 0 }
    messages[1] = { role: "assistant", content: [{ type: "text", text: "needle one, needle two" }], provider: "anthropic", model: "claude", stopReason: "stop", timestamp: 1 }
    render(<Harness transcript={transcriptOf(messages)} />)
    expect(screen.queryByText("Answer 0")).toBeNull()

    act(() => focusPiSearchHit("s1", "needle", hit("e1", 1), [hit("e0"), hit("e1", 1)]))

    // Turn 0 was out of the window and collapsed: now mounted and open, its answer text rendered
    expect(screen.getByText(/needle one, needle two/)).toBeTruthy()
    expect(scrolled.at(-1)?.textContent).toContain("needle one, needle two")
    expect(painted("pi-search")).toEqual(["needle", "needle", "needle"])
    expect(painted("pi-search-current")).toHaveLength(1)
    expect(registry.get("pi-search-current")?.ranges[0].startContainer.textContent).toContain("needle one, needle two")
    expect(screen.getByText("2 / 2")).toBeTruthy()
  })

  it("steps through the hits with Enter and Shift+Enter, clears on Escape, and leaves text fields alone", () => {
    const messages = history(2)
    messages[0] = { role: "user", content: "needle", timestamp: 0 }
    messages[3] = { role: "assistant", content: [{ type: "text", text: "a needle" }], provider: "anthropic", model: "claude", stopReason: "stop", timestamp: 3 }
    render(<><Harness transcript={transcriptOf(messages)} /><textarea aria-label="composer" /></>)
    act(() => focusPiSearchHit("s1", "needle", hit("e0"), [hit("e0"), hit("e3")]))
    expect(screen.getByText("1 / 2")).toBeTruthy()

    fireEvent.keyDown(document.body, { key: "Enter" })
    expect(screen.getByText("2 / 2")).toBeTruthy()
    fireEvent.keyDown(document.body, { key: "Enter", shiftKey: true })
    fireEvent.keyDown(document.body, { key: "Enter", shiftKey: true })
    expect(screen.getByText("2 / 2")).toBeTruthy()
    fireEvent.keyDown(screen.getByLabelText("composer"), { key: "Enter" })
    expect(screen.getByText("2 / 2")).toBeTruthy()

    fireEvent.keyDown(document.body, { key: "Escape" })
    expect(screen.queryByText("2 / 2")).toBeNull()
    expect(registry.has("pi-search")).toBe(false)
    expect(registry.has("pi-search-current")).toBe(false)
  })

  it("drops a hit whose entry is not on the shown branch", () => {
    render(<Harness transcript={transcriptOf(history(2))} />)
    act(() => focusPiSearchHit("s1", "answer", hit("other")))
    expect(scrolled).toEqual([])
  })
})
