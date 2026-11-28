import { act, cleanup, renderHook } from "@testing-library/react"
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest"
import { liveSession, liveStatus, observeLiveSessions } from "../liveSession"
import { clearOutbox, heldOf } from "../outbox"
import { usePiSession, usePiTranscript } from "../usePiSession"

/** pi-host's end of the page's socket: the real `piSocket` talks to it. */
class FakeSocket {
  static OPEN = 1
  static instances: FakeSocket[] = []
  readyState = 0
  sent: { type: string; sessionId?: string; cursor?: string | null; command?: { id: string; type: string; message?: string } }[] = []
  onopen: (() => void) | null = null
  onmessage: ((message: { data: string }) => void) | null = null
  onclose: ((event: { reason: string }) => void) | null = null

  constructor() {
    FakeSocket.instances.push(this)
  }

  send(data: string) {
    this.sent.push(JSON.parse(data))
  }

  close() {}

  open() {
    if (this.readyState === FakeSocket.OPEN) return
    this.readyState = FakeSocket.OPEN
    this.onopen?.()
  }

  emit(frame: object) {
    act(() => this.onmessage?.({ data: JSON.stringify(frame) }))
  }

  drop() {
    this.readyState = 3
    act(() => this.onclose?.({ reason: "gone" }))
  }
}

function host(): FakeSocket {
  const current = FakeSocket.instances.at(-1)!
  current.open()
  return current
}

const prompt = { type: "message", id: "e1", parentId: null, timestamp: "2026-01-01T00:00:00Z", message: { role: "user", content: "Hi", timestamp: 1 } }

function attached(sessionId: string, running = false) {
  return {
    type: "attached", sessionId, streamId: "x", sequence: 0,
    state: { isStreaming: running, sessionId, model: null, thinkingLevel: "off" },
    entries: { entries: [prompt], leafId: "e1" }, ui: { dialogs: [], statuses: {}, queue: { steering: [], followUp: [] } },
  }
}

const framesOf = (socket: FakeSocket, sessionId: string, type: string) => socket.sent.filter((frame) => frame.sessionId === sessionId && frame.type === type)

beforeAll(() => {
  vi.stubGlobal("WebSocket", FakeSocket)
})

afterEach(() => {
  cleanup()
  clearOutbox()
})

describe("a live session", () => {
  it("attaches once for two holders, the last release detaches and empties its state", () => {
    const session = liveSession("s1")
    const releaseFirst = session.attach()
    const releaseSecond = session.attach()
    const socket = host()
    socket.emit(attached("s1"))
    expect(framesOf(socket, "s1", "attach")).toHaveLength(1)
    expect(session.state.get().status).toEqual({ kind: "ready" })
    expect(session.transcript.get().messages).toHaveLength(1)

    releaseFirst()
    expect(framesOf(socket, "s1", "detach")).toHaveLength(0)
    releaseSecond()
    expect(framesOf(socket, "s1", "detach")).toHaveLength(1)
    expect(session.state.get().status).toEqual({ kind: "connecting" })
    expect(session.transcript.get().messages).toEqual([])
  })

  it("resyncs from the cursor once after a reconnect", async () => {
    const session = liveSession("s2")
    const release = session.attach()
    const first = host()
    first.emit(attached("s2"))
    first.drop()
    // piSocket reconnects after its backoff
    await vi.waitFor(() => expect(FakeSocket.instances.at(-1)).not.toBe(first))
    const second = host()
    expect(framesOf(second, "s2", "resync")).toEqual([{ type: "resync", sessionId: "s2", cursor: "e1" }])
    expect(framesOf(second, "s2", "attach")).toEqual([])
    release()
  })

  it("fails pending commands on a drop and shows a dispatched held message as not sent", async () => {
    const session = liveSession("s3")
    const release = session.attach()
    const socket = host()
    socket.emit(attached("s3", true))
    let answer: Promise<void> | undefined
    await act(async () => {
      answer = session.compact()
      await session.send("later", [], "followUp")
    })
    expect(heldOf("s3")).toMatchObject([{ text: "later", dispatched: true, state: "sending" }])
    socket.drop()
    await act(async () => { await answer })
    expect(heldOf("s3")).toMatchObject([{ text: "later", state: "failed" }])
    expect(session.state.get().notices).toMatchObject([{ text: "Disconnected", tone: "error" }])
    release()
  })

  it("drops what its pending commands report after the last release", async () => {
    const session = liveSession("s10")
    const release = session.attach()
    const socket = host()
    socket.emit(attached("s10"))
    void session.send("hi", [])
    void session.compact()
    const sent = socket.sent.length
    release()
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)) })
    expect(socket.sent.slice(sent).filter((frame) => frame.type === "command")).toEqual([])
    const releaseAgain = session.attach()
    expect(session.state.get().notices).toEqual([])
    releaseAgain()
  })

  it("resyncs each attached session once when the tab turns visible", () => {
    const releases = ["s11", "s12"].map((id) => liveSession(id).attach())
    const socket = host()
    for (const id of ["s11", "s12"]) socket.emit(attached(id))
    act(() => { document.dispatchEvent(new Event("visibilitychange")) })
    expect(["s11", "s12"].map((id) => framesOf(socket, id, "resync").length)).toEqual([1, 1])
    releases.forEach((release) => release())
  })

  it("tells status observers only of what liveStatus reads (measure)", () => {
    const onStatus = vi.fn()
    const stop = observeLiveSessions({ onStatus, onEvent: vi.fn() })
    const release = liveSession("s13").attach()
    const socket = host()
    socket.emit(attached("s13"))
    onStatus.mockClear()
    const request = (fields: object) => socket.emit({ type: "pi", sessionId: "s13", record: { type: "extension_ui_request", ...fields } })
    request({ id: "w1", method: "setWidget", widgetKey: "k", widgetLines: ["busy"] })
    // Was 1: every patch reached the observers
    expect(onStatus).toHaveBeenCalledTimes(0)
    request({ id: "d1", method: "confirm", title: "Ok?" })
    expect(onStatus).toHaveBeenCalledTimes(1)
    stop()
    release()
  })
})

describe("two views of one session", () => {
  it("both show the session, the second mounted after it attached (measure)", () => {
    const useView = () => ({ status: usePiSession("s4", vi.fn()).status, transcript: usePiTranscript("s4") })
    const first = renderHook(useView)
    const socket = host()
    socket.emit(attached("s4"))
    const second = renderHook(useView)
    const ready = [first, second].filter((view) => view.result.current.status.kind === "ready").length
    // Was 1: the second view got no `attached` frame and stayed "connecting"
    expect(ready).toBe(2)
    expect(second.result.current.transcript).toBe(first.result.current.transcript)
    expect(framesOf(socket, "s4", "attach")).toHaveLength(1)
  })
})

describe("the session's stats", () => {
  it("refresh after every assistant message, not only once the run settles", async () => {
    const view = renderHook(() => usePiSession("s5", vi.fn()))
    const socket = host()
    socket.emit(attached("s5", true))
    const answerStats = async (cost: number) => {
      const request = socket.sent.findLast((frame) => frame.sessionId === "s5" && frame.command?.type === "get_session_stats")!
      await act(async () => socket.onmessage?.({ data: JSON.stringify({ type: "pi", sessionId: "s5", record: { type: "response", id: request.command!.id, success: true, data: { cost } } }) }))
    }
    await answerStats(0.1)
    expect(view.result.current.stats).toEqual({ cost: 0.1 })

    socket.emit({ type: "pi", sessionId: "s5", record: { type: "message_end", message: { role: "assistant", content: [], stopReason: "toolUse", timestamp: 2 } } })
    await answerStats(0.2)
    expect(view.result.current.stats).toEqual({ cost: 0.2 })
  })
})

describe("a live session's run state", () => {
  const piRecord = (sessionId: string, record: object) => ({ type: "pi", sessionId, record })
  const delta = (text: string) => ({ type: "message_update", assistantMessageEvent: { type: "text_delta", contentIndex: 0, delta: text } })

  it("is not running or waiting after pi exits", () => {
    const session = liveSession("s6")
    const release = session.attach()
    const socket = host()
    socket.emit({ ...attached("s6", true), ui: { dialogs: [{ type: "extension_ui_request", id: "d1", method: "confirm", title: "Ok?" }], statuses: {}, queue: { steering: [], followUp: [] } } })
    expect(liveStatus("s6")).toMatchObject({ running: true, waiting: true })
    socket.emit({ type: "exit", sessionId: "s6", code: 1 })
    expect(liveStatus("s6")).toMatchObject({ running: false, waiting: false })
    release()
  })

  describe("with animation frames under test control", () => {
    let frames: (() => void)[] = []
    let clock = 0
    const runFrame = () => {
      const due = frames
      frames = []
      clock += 100
      act(() => due.forEach((frame) => frame()))
    }
    beforeAll(() => {
      vi.stubGlobal("requestAnimationFrame", (callback: (now: number) => void) => frames.push(() => callback(clock)))
      vi.stubGlobal("cancelAnimationFrame", (id: number) => { frames[id - 1] = () => {} })
    })

    const streamingSession = (id: string) => {
      frames = []
      const session = liveSession(id)
      const release = session.attach()
      const socket = host()
      socket.emit(attached(id, true))
      socket.emit(piRecord(id, { type: "agent_start" }))
      socket.emit(piRecord(id, { type: "message_start", message: { role: "assistant", content: [], timestamp: 2 } }))
      runFrame()
      return { session, release, socket }
    }

    it("applies a burst of deltas once per frame, then the message end", () => {
      const { session, release, socket } = streamingSession("s7")
      const seen: unknown[] = []
      session.transcript.subscribe(() => seen.push(session.transcript.get()))
      for (const text of ["a", "b", "c", "d", "e"]) socket.emit(piRecord("s7", delta(text)))
      expect(seen).toHaveLength(0)
      socket.emit(piRecord("s7", { type: "message_end", message: { role: "assistant", content: [{ type: "text", text: "abcde" }], stopReason: "stop", timestamp: 2 } }))
      expect(seen).toHaveLength(2)
      expect(seen[0]).not.toBe(seen[1])
      release()
    })

    it("drops buffered deltas when the last holder releases", () => {
      const { session, release, socket } = streamingSession("s8")
      socket.emit(piRecord("s8", delta("lost")))
      release()
      runFrame()
      expect(session.transcript.get().messages).toEqual([])
    })

    it("emits settled once per run, after the transcript stopped running", () => {
      const { release, socket } = streamingSession("s9")
      const seen: unknown[] = []
      const stop = observeLiveSessions({ onStatus: () => {}, onEvent: (event) => event.type === "settled" && seen.push(liveStatus("s9")?.running) })
      socket.emit(piRecord("s9", { type: "agent_end", messages: [] }))
      socket.emit(piRecord("s9", { type: "agent_settled" }))
      socket.emit(piRecord("s9", { type: "agent_settled" }))
      expect(seen).toEqual([false])
      stop()
      release()
    })
  })
})
