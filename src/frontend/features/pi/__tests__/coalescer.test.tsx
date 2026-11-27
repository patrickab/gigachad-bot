import type { Dispatch } from "react"
import { describe, expect, it, vi } from "vitest"
import { createTranscriptCoalescer, scheduleAtDisplayRate, type FlushScheduler } from "../coalescer"
import type { TranscriptAction } from "../transcript"
import type { AssistantMessage } from "../types"

/** A scheduler the test fires by hand, standing in for `requestAnimationFrame`. */
function manualFrames() {
  let pending: (() => void) | null = null
  const schedule: FlushScheduler = (flush) => {
    pending = flush
    return () => { pending = null }
  }
  return { schedule, frame: () => { const flush = pending; pending = null; flush?.() }, scheduled: () => pending !== null }
}

const reply: AssistantMessage = { role: "assistant", content: [], provider: "anthropic", model: "claude", stopReason: "stop", timestamp: 2_000 }
const delta = (text: string): TranscriptAction => ({ kind: "event", event: { type: "message_update", assistantMessageEvent: { type: "text_delta", contentIndex: 0, delta: text } } })

describe("createTranscriptCoalescer", () => {
  it("buffers deltas and tool updates into one events action per frame", () => {
    const frames = manualFrames()
    const dispatch = vi.fn<Dispatch<TranscriptAction>>()
    const coalescer = createTranscriptCoalescer(dispatch, frames.schedule)
    coalescer.push(delta("a"))
    coalescer.push({ kind: "event", event: { type: "tool_execution_update", toolCallId: "call-1" } })
    coalescer.push(delta("b"))
    expect(dispatch).not.toHaveBeenCalled()
    frames.frame()
    expect(dispatch).toHaveBeenCalledTimes(1)
    expect(dispatch.mock.calls[0][0]).toMatchObject({ kind: "events", events: [{ type: "message_update" }, { type: "tool_execution_update" }, { type: "message_update" }] })
  })

  it("flushes the buffer before any other action so order is kept, and drops it on reset", () => {
    const frames = manualFrames()
    const dispatch = vi.fn<Dispatch<TranscriptAction>>()
    const coalescer = createTranscriptCoalescer(dispatch, frames.schedule)
    coalescer.push(delta("a"))
    coalescer.push({ kind: "event", event: { type: "message_end", message: reply } })
    expect(dispatch.mock.calls.map(([action]) => action.kind === "event" ? action.event.type : action.kind)).toEqual(["events", "message_end"])
    expect(frames.scheduled()).toBe(false)

    coalescer.push(delta("b"))
    coalescer.reset()
    frames.frame()
    expect(dispatch).toHaveBeenCalledTimes(2)
  })
})

describe("scheduleAtDisplayRate", () => {
  it("skips a frame that comes under 12 ms after the last flush, so a 144 Hz display flushes every other frame", () => {
    const frames: FrameRequestCallback[] = []
    vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => frames.push(callback))
    vi.stubGlobal("cancelAnimationFrame", () => {})
    const flush = vi.fn()
    const frame = (now: number) => frames.shift()?.(now)

    scheduleAtDisplayRate(flush)
    frame(1000)
    expect(flush).toHaveBeenCalledTimes(1)
    scheduleAtDisplayRate(flush)
    frame(1007)
    expect(flush).toHaveBeenCalledTimes(1)
    frame(1014)
    expect(flush).toHaveBeenCalledTimes(2)
    vi.unstubAllGlobals()
  })
})
