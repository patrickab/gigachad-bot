import { act, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { PiTranscript } from "../PiTranscript"
// The chat's tool renderers register under names Pi sessions use too: they must not change what folds
import "@/features/chat/toolRenderers"
import { clearPiSearchHit, focusPiSearchHit, usePiSearchFocus } from "../searchFocus"
import type { Attachment } from "@/lib/domain/types"
import { EMPTY_TRANSCRIPT, transcriptReducer, type SessionEntry, type Transcript } from "../transcript"
import type { AgentMessage, AssistantMessage } from "../types"

// The hits of the whole session are read through the API: a failure keeps the listed ones
vi.mock("@/lib/backend/api", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/lib/backend/api")>(),
  fetchPiSessionMatches: () => Promise.reject(new Error("offline")),
}))

const preview = vi.hoisted(() => ({ shown: null as Attachment | null }))
vi.mock("@/features/chat/preview/AttachmentPreview", () => ({
  AttachmentPreview: ({ attachment }: { attachment: Attachment | null }) => {
    preview.shown = attachment
    return attachment ? <div data-testid="lightbox">{attachment.url}</div> : null
  },
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
  it("offers branching from a settled turn's last persisted entry", () => {
    const onBranch = vi.fn()
    render(<PiTranscript transcript={transcriptOf(history(2))} fileLinks={fileLinks} onBranch={onBranch} />)
    fireEvent.click(screen.getByRole("button", { name: "Branch session from this point" }))
    expect(onBranch).toHaveBeenCalledWith("e3")
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
})

describe("PiTranscript tool images and reads", () => {
  const HASH = "a".repeat(64)
  const call = (id: string, name: string, args: Record<string, unknown>): AgentMessage => ({ role: "assistant", content: [{ type: "toolCall", id, name, arguments: args }], provider: "anthropic", model: "claude", stopReason: "toolUse", timestamp: 2 })
  const result = (id: string, name: string, content: unknown[]): AgentMessage => ({ role: "toolResult", toolCallId: id, toolName: name, content, isError: false, timestamp: 3 } as AgentMessage)
  const done: AgentMessage = { role: "assistant", content: [{ type: "text", text: "Done" }], provider: "anthropic", model: "claude", stopReason: "stop", timestamp: 4 }

  it("shows a settled tool's image as an attachment outside the fold and opens it in the lightbox", () => {
    const messages: AgentMessage[] = [
      { role: "user", content: "Screenshot it", timestamp: 1 },
      call("t1", "screenshot", {}),
      result("t1", "screenshot", [{ type: "text", text: "ok" }, { type: "image", mimeType: "image/png", hash: HASH }]),
      done,
    ]
    render(<PiTranscript transcript={transcriptOf(messages)} fileLinks={fileLinks} />)

    const thumb = screen.getByRole("img", { name: `screenshot-${HASH.slice(0, 8)}.png` })
    expect(thumb.getAttribute("src")).toMatch(new RegExp(`/pi/media/${HASH}/thumb$`))
    // The fold is closed, the image is not inside it
    expect(screen.getByRole("button", { name: /Process details/ }).getAttribute("aria-expanded")).not.toBe("true")
    fireEvent.click(screen.getByRole("button", { name: `screenshot-${HASH.slice(0, 8)}.png` }))
    expect(screen.getByTestId("lightbox").textContent).toMatch(new RegExp(`/pi/media/${HASH}$`))
  })

  it("shows an inline image, one that pi-host could not store, from its data URL", () => {
    const messages: AgentMessage[] = [
      { role: "user", content: "Plot", timestamp: 1 },
      call("t1", "plot", {}),
      result("t1", "plot", [{ type: "image", mimeType: "image/png", data: "QUJD" }]),
      done,
    ]
    render(<PiTranscript transcript={transcriptOf(messages)} fileLinks={fileLinks} />)
    expect(screen.getByRole("img", { name: "plot-1.png" }).getAttribute("src")).toBe("data:image/png;base64,QUJD")
  })

  it("shows a live tool's image from its running result", () => {
    const messages: AgentMessage[] = [{ role: "user", content: "Look", timestamp: 1 }, call("t1", "read", { path: "a.png" })]
    const runs = { t1: { running: true, isError: false, result: { content: [{ type: "image" as const, mimeType: "image/png", hash: HASH }] } } }
    render(<PiTranscript transcript={transcriptOf(messages, { running: true, runs })} fileLinks={fileLinks} />)
    expect(screen.getByRole("img", { name: `read-${HASH.slice(0, 8)}.png` })).toBeTruthy()
  })
})

describe("PiTranscript user images", () => {
  const reply: AgentMessage = { role: "assistant", content: [{ type: "text", text: "Seen" }], provider: "anthropic", model: "claude", stopReason: "stop", timestamp: 2 }

  it("shows a user message's image block as a chat attachment", () => {
    const content = [{ type: "text" as const, text: "Look" }, { type: "image" as const, mimeType: "image/png", data: "QUJD" }]
    render(<PiTranscript transcript={transcriptOf([{ role: "user", content, timestamp: 1 }, reply])} fileLinks={fileLinks} />)
    expect(screen.getByRole("img", { name: "image-1.png" }).getAttribute("src")).toBe("data:image/png;base64,QUJD")
  })

  it("shows the image of an optimistic prompt", () => {
    const optimistic: AgentMessage = { role: "user", content: [{ type: "text", text: "Look" }, { type: "image", mimeType: "image/png", data: "QUJD" }], timestamp: 1 }
    render(<PiTranscript transcript={transcriptOf([], { optimistic: optimistic as never })} fileLinks={fileLinks} />)
    expect(screen.getByRole("img", { name: "image-1.png" })).toBeTruthy()
  })
})
