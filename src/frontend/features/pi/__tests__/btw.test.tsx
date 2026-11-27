import { afterEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, fireEvent, render, renderHook, screen } from "@testing-library/react"
import type { SessionSubscriber } from "../piSocket"
import { clearOutbox } from "../outbox"
import { BtwPanel } from "../BtwPanel"
import { parseBtwWidget, type BtwThread } from "../btw"
import { usePiSession } from "../usePiSession"

const socket = vi.hoisted(() => ({ subscriber: null as SessionSubscriber | null, sent: [] as Record<string, unknown>[] }))
vi.mock("../piSocket", () => ({
  subscribePiSession: (_id: string, subscriber: SessionSubscriber) => {
    socket.subscriber = subscriber
    return () => {}
  },
  sendPiCommand: (_id: string, command: Record<string, unknown>) => {
    socket.sent.push(command)
    return true
  },
  sendPiFrame: () => true,
}))
vi.mock("@/components/content/LaTeXMarkdown", () => ({ LaTeXMarkdown: ({ content }: { content: string }) => <div>{content}</div> }))

const thread = (over: Partial<BtwThread> = {}): BtwThread => ({ id: "btw-1", question: "why?", answer: "because", status: "done", followUps: [], createdAt: "2026-01-01T00:00:00Z", ...over })

afterEach(() => {
  cleanup()
  socket.subscriber = null
  socket.sent.length = 0
  clearOutbox()
})

describe("btw state", () => {
  it("parses threads, none without lines, null for unknown versions", () => {
    expect(parseBtwWidget({})).toEqual([])
    expect(parseBtwWidget({ widgetLines: [JSON.stringify({ type: "btw_state", version: 1, state: { threads: [thread()] } })] })).toEqual([thread()])
    expect(parseBtwWidget({ widgetLines: [JSON.stringify({ type: "btw_state", version: 2, state: { threads: [] } })] })).toBeNull()
    expect(parseBtwWidget({ widgetLines: ["nope"] })).toBeNull()
  })
})

describe("usePiSession btw", () => {
  const attach = (running: boolean) => act(() => socket.subscriber!.onFrame({
    type: "attached", sessionId: "s1", streamId: "x", sequence: 0,
    state: { isStreaming: running, sessionId: "s1", model: null, thinkingLevel: "off" },
    entries: { entries: [], leafId: null }, ui: { dialogs: [], statuses: {}, queue: { steering: [], followUp: [] } },
  }))

  it("sends btw commands directly mid-run, never held", async () => {
    const { result } = renderHook(() => usePiSession("s1", vi.fn()))
    attach(true)
    socket.sent.length = 0
    await act(async () => {
      void result.current.send("/btw what now?", [])
      result.current.btw.cancel("btw-1")
    })
    expect(socket.sent.map(({ type, message, streamingBehavior }) => ({ type, message, streamingBehavior }))).toEqual([
      { type: "prompt", message: "/btw what now?", streamingBehavior: undefined },
      { type: "prompt", message: "/btw-cancel btw-1", streamingBehavior: undefined },
    ])
    expect(result.current.held).toEqual([])
  })
})

describe("BtwPanel", () => {
  const actions = () => ({ followUp: vi.fn(), cancel: vi.fn(), dismiss: vi.fn() })

  it("shows the newest thread, follows up with Enter, closes with Esc and x", () => {
    const a = actions()
    render(<BtwPanel threads={[thread({ id: "old", question: "older q" }), thread({ id: "btw-2", question: "newest q", followUps: [{ question: "and?", answer: "more", status: "done", createdAt: "t" }] })]} actions={a} />)
    expect(screen.getByText("newest q")).toBeTruthy()
    expect(screen.getByText("more")).toBeTruthy()
    const input = screen.getByRole("textbox", { name: "Follow-up" })
    fireEvent.change(input, { target: { value: "next" } })
    fireEvent.keyDown(input, { key: "Enter" })
    expect(a.followUp).toHaveBeenCalledWith("btw-2", "next")
    fireEvent.keyDown(input, { key: "Escape" })
    expect(a.dismiss).toHaveBeenCalledWith("btw-2")
    fireEvent.click(screen.getByRole("button", { name: "Close side question" }))
    expect(a.dismiss).toHaveBeenCalledTimes(2)
    expect(screen.getByText("Earlier (1)")).toBeTruthy()
  })
})
