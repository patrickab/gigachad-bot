import { fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { PiTranscript, TURN_PAGE_SIZE } from "../PiTranscript"
import { EMPTY_TRANSCRIPT, type Transcript } from "../transcript"
import type { AgentMessage, AssistantMessage } from "../types"

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
})
