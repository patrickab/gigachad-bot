import { act, render } from "@testing-library/react"
import { useReducer, type Dispatch } from "react"
import { describe, expect, it, vi } from "vitest"
import { createTranscriptCoalescer, scheduleAtDisplayRate, useCoalescedDispatch, type FlushScheduler } from "../coalescer"
import { EMPTY_TRANSCRIPT, transcriptReducer, type Transcript, type TranscriptAction } from "../transcript"
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

describe("useCoalescedDispatch", () => {
  it("renders once per frame for a burst of deltas", () => {
    const frames = manualFrames()
    let renders = 0
    let push: Dispatch<TranscriptAction> = () => {}
    let shown: Transcript = EMPTY_TRANSCRIPT
    function Probe() {
      const [transcript, dispatch] = useReducer(transcriptReducer, EMPTY_TRANSCRIPT)
      push = useCoalescedDispatch(dispatch, frames.schedule)
      shown = transcript
      renders++
      return null
    }
    render(<Probe />)
    act(() => push({ kind: "event", event: { type: "message_start", message: reply } }))
    act(() => push({ kind: "event", event: { type: "message_update", assistantMessageEvent: { type: "text_start", contentIndex: 0 } } }))
    const before = renders
    act(() => {
      for (let index = 0; index < 50; index++) push(delta("x"))
    })
    expect(renders).toBe(before)
    act(() => frames.frame())
    expect(renders).toBe(before + 1)
    expect(shown.streaming?.content).toEqual([{ type: "text", text: "x".repeat(50) }])
  })
})

describe("message_update wire size", () => {
  it("stays constant per frame (delta-only), where resending the whole message grows with the reply", () => {
    const bytes = (value: unknown) => JSON.stringify(value).length
    const deltaFrame = (text: string) => bytes({ type: "message_update", assistantMessageEvent: { type: "text_delta", contentIndex: 0, delta: text } })
    const wholeFrame = (so_far: string) => bytes({ type: "message_update", message: { role: "assistant", content: [{ type: "text", text: so_far }] }, assistantMessageEvent: { type: "text_delta", contentIndex: 0, delta: "word " } })
    const words = Array.from({ length: 1000 }, () => "word ").join("")
    // Before (full message per frame): ~5 KB at word 1000. Delta frame: ~100 B, independent of position.
    expect(deltaFrame("word ")).toBe(deltaFrame("word "))
    expect(deltaFrame("word ")).toBeLessThan(120)
    expect(wholeFrame(words)).toBeGreaterThan(40 * deltaFrame("word "))
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
