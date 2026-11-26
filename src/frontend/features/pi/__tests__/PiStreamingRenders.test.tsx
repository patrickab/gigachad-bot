import { act, cleanup, fireEvent, render, screen } from "@testing-library/react"
import { useReducer } from "react"
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest"
import { PiTranscript } from "../PiTranscript"
import { EMPTY_TRANSCRIPT, transcriptReducer, type TranscriptAction } from "../transcript"
import type { AgentMessage } from "../types"

/**
 * A streamed delta re-renders the live turn only, wired as `PiTab` wires the transcript: the session's reducer, stable
 * callbacks and file links. Text is counted where it renders (`LaTeXMarkdown`), for an open settled turn and a collapsed one.
 */

const renders = vi.hoisted(() => new Map<string, number>())
vi.mock("@/components/content/LaTeXMarkdown", () => ({
  LaTeXMarkdown: ({ content }: { content: string }) => {
    renders.set(content, (renders.get(content) ?? 0) + 1)
    return <div>{content}</div>
  },
}))

beforeAll(() => { Element.prototype.scrollIntoView = () => {} })
vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} })

const settled: AgentMessage[] = [0, 1, 2].flatMap((index): AgentMessage[] => [
  { role: "user", content: `Prompt ${index}`, timestamp: index * 10 },
  { role: "assistant", content: [{ type: "text", text: `Answer ${index}` }], provider: "anthropic", model: "claude", stopReason: "stop", timestamp: index * 10 + 1 },
])
const initial = { ...EMPTY_TRANSCRIPT, messages: settled, entryIds: settled.map((_, index) => `e${index}`) }

const fileLinks = { base: "/repo", open: () => {} }
const noop = () => {}

function Session({ onDispatch }: { onDispatch: (dispatch: (action: TranscriptAction) => void) => void }) {
  const [transcript, dispatch] = useReducer(transcriptReducer, initial)
  onDispatch(dispatch)
  return <PiTranscript transcript={transcript} held={[]} onRetryHeld={noop} onDeleteHeld={noop} fileLinks={fileLinks} onBranch={noop} onColumnChange={noop} />
}

const event = (record: Record<string, unknown>): TranscriptAction => ({ kind: "event", event: { type: String(record.type), ...record } })
const delta = (text: string) => event({ type: "message_update", assistantMessageEvent: { type: "text_delta", contentIndex: 0, delta: text } })

describe("a Pi session while its answer streams", () => {
  afterEach(cleanup)

  it("re-renders neither an open nor a collapsed settled turn", () => {
    let dispatch!: (action: TranscriptAction) => void
    render(<Session onDispatch={(current) => { dispatch = current }} />)
    // Prompt 0 open, Prompt 1 collapsed to its answer (rendered at once: jsdom has no IntersectionObserver)
    fireEvent.click(screen.getByRole("button", { name: "Prompt 0" }))

    act(() => dispatch({ kind: "prompt", prompt: { role: "user", content: "Prompt 3", timestamp: 100 } }))
    act(() => dispatch(event({ type: "agent_start" })))
    act(() => dispatch(event({ type: "message_start", message: { role: "assistant", content: [], provider: "anthropic", model: "claude", stopReason: "pending", timestamp: 101 } })))
    act(() => dispatch(event({ type: "message_update", assistantMessageEvent: { type: "text_start", contentIndex: 0 } })))
    act(() => dispatch(delta("A")))
    expect(renders.get("A")).toBeGreaterThan(0)

    const watched = ["Prompt 0", "Answer 0", "Answer 1"]
    const before = watched.map((key) => renders.get(key) ?? 0)
    expect(before.every((count) => count > 0)).toBe(true)

    let answer = "A"
    for (const token of ["n", "s", "w", "e", "r"]) {
      answer += token
      act(() => dispatch(delta(token)))
      expect(renders.get(answer)).toBeGreaterThan(0)
    }
    expect(watched.map((key) => renders.get(key) ?? 0)).toEqual(before)
  })
})
