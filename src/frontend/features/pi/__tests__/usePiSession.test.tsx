import { afterEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, renderHook } from "@testing-library/react"
import type { HostFrame, SessionSubscriber } from "../piSocket"
import { clearOutbox } from "../outbox"
import { OPEN_FILE_NOTICE_PREFIX, usePiSession } from "../usePiSession"

const socket = vi.hoisted(() => ({ subscriber: null as SessionSubscriber | null, connected: true, sent: [] as Record<string, unknown>[], onSend: undefined as ((command: Record<string, unknown>) => void) | undefined }))
vi.mock("../piSocket", () => ({
  subscribePiSession: (_sessionId: string, subscriber: SessionSubscriber) => {
    socket.subscriber = subscriber
    return () => {}
  },
  sendPiCommand: (_sessionId: string, command: Record<string, unknown>) => {
    if (!socket.connected) return false
    socket.sent.push(command)
    socket.onSend?.(command)
    return true
  },
  sendPiFrame: () => true,
}))

function notifyFrame(message: string): HostFrame {
  return { type: "pi", sessionId: "s1", record: { type: "extension_ui_request", id: "u1", method: "notify", message, notifyType: "info" } }
}

describe("usePiSession notify", () => {
  afterEach(() => {
    cleanup()
    socket.subscriber = null
    socket.onSend = undefined
  })

  it("opens an open-file notice's path instead of showing it", () => {
    const onOpenFile = vi.fn()
    const { result } = renderHook(() => usePiSession("s1", onOpenFile))
    act(() => socket.subscriber!.onFrame(notifyFrame(`${OPEN_FILE_NOTICE_PREFIX}/repo/src/app.ts`)))
    expect(onOpenFile).toHaveBeenCalledWith("/repo/src/app.ts")
    expect(result.current.notices).toEqual([])
  })

  it("shows any other notice", () => {
    const onOpenFile = vi.fn()
    const { result } = renderHook(() => usePiSession("s1", onOpenFile))
    act(() => socket.subscriber!.onFrame(notifyFrame("Saved")))
    expect(onOpenFile).not.toHaveBeenCalled()
    expect(result.current.notices).toMatchObject([{ text: "Saved", tone: "info" }])
  })
})

describe("usePiSession held messages", () => {
  afterEach(() => {
    cleanup()
    socket.subscriber = null
    socket.connected = true
    socket.sent.length = 0
    clearOutbox()
  })

  const attach = (running = true) => act(() => socket.subscriber!.onFrame({
    type: "attached", sessionId: "s1", streamId: "x", sequence: 0,
    state: { isStreaming: running, sessionId: "s1", model: null, thinkingLevel: "off" },
    entries: { entries: [], leafId: null }, ui: { dialogs: [], statuses: {}, queue: { steering: [], followUp: [] } },
  }))
  const respond = (id: string, success: boolean, data?: unknown) => act(() => socket.subscriber!.onFrame({ type: "pi", sessionId: "s1", record: { type: "response", id, success, data, error: success ? undefined : "no" } }))
  const prompts = () => socket.sent.filter((command) => command.type === "prompt")

  it("holds a message during a run, queued once Pi accepts it, gone when the queue shows it", async () => {
    const { result } = renderHook(() => usePiSession("s1", vi.fn()))
    attach()
    await act(async () => { await result.current.send("steer me", [], "steer") })
    expect(prompts()[0]).toMatchObject({ message: "steer me", streamingBehavior: "steer" })
    expect(result.current.held[0]).toMatchObject({ text: "steer me", state: "sending" })
    await act(async () => respond(prompts()[0].id as string, true))
    expect(result.current.held[0].state).toBe("queued")
    act(() => socket.subscriber!.onFrame({ type: "pi", sessionId: "s1", record: { type: "queue_update", steering: ["steer me"], followUp: [] } }))
    expect(result.current.held).toEqual([])
  })

  it("keeps messages held while the socket is down and flushes them in order once attached", async () => {
    socket.connected = false
    const { result } = renderHook(() => usePiSession("s1", vi.fn()))
    attach()
    await act(async () => {
      await result.current.send("first", [], "steer")
      await result.current.send("second", [], "followUp")
    })
    expect(prompts()).toEqual([])
    expect(result.current.held.map((entry) => entry.text)).toEqual(["first", "second"])
    act(() => socket.subscriber!.onStatus({ kind: "connecting" }))
    socket.connected = true
    attach()
    expect(prompts().map((command) => [command.message, command.streamingBehavior])).toEqual([["first", "steer"], ["second", "followUp"]])
  })

  it("shows a dropped delivery as not sent and does not resend it, Retry does", async () => {
    const { result } = renderHook(() => usePiSession("s1", vi.fn()))
    attach()
    await act(async () => { await result.current.send("maybe", [], "steer") })
    await act(async () => socket.subscriber!.onStatus({ kind: "connecting" }))
    expect(result.current.held[0].state).toBe("failed")
    attach()
    expect(prompts()).toHaveLength(1)
    act(() => result.current.retryHeld(result.current.held[0].id))
    expect(prompts()).toHaveLength(2)
  })

  it("abort returns held and queued messages to the draft", async () => {
    const { result } = renderHook(() => usePiSession("s1", vi.fn()))
    attach()
    await act(async () => { await result.current.send("a", [], "followUp") })
    let back: string[] = []
    await act(async () => {
      const aborting = result.current.abort()
      await Promise.resolve()
      const clear = socket.sent.find((command) => command.type === "clear_queue")!
      respond(clear.id as string, true, { steering: [], followUp: ["a"] })
      await Promise.resolve()
      const abort = socket.sent.find((command) => command.type === "abort")!
      respond(abort.id as string, true)
      back = await aborting
    })
    expect(back).toEqual(["a"])
    expect(result.current.held).toEqual([])
  })
})

function widgetFrame(widgetKey: string, widgetLines?: string[]): HostFrame {
  return { type: "pi", sessionId: "s1", record: { type: "extension_ui_request", id: "w1", method: "setWidget", widgetKey, widgetLines } }
}

describe("usePiSession subagent widgets", () => {
  afterEach(() => {
    cleanup()
    socket.subscriber = null
    socket.onSend = undefined
    socket.connected = true
  })

  it("keeps text widgets by key, a cleared one goes", () => {
    const { result } = renderHook(() => usePiSession("s1", vi.fn()))
    act(() => socket.subscriber!.onFrame(widgetFrame("subagent-async", ["PI_SUBAGENT_ASYNC_JSON:{}"])))
    expect(result.current.widgets).toEqual({ "subagent-async": ["PI_SUBAGENT_ASYNC_JSON:{}"] })
    act(() => socket.subscriber!.onFrame(widgetFrame("subagent-async")))
    expect(result.current.widgets).toEqual({})
  })

  it("answers an inspect request from the transient reply widget without keeping it", async () => {
    let requestId = ""
    socket.onSend = (command) => {
      if (command.type === "prompt") {
        const [name, id, asyncId, childId] = String(command.message).split(" ")
        expect([name, asyncId, childId]).toEqual(["/subagents-inspect-rpc", "run1", "step1"])
        requestId = id!
        // The command answers first, then emits the reply
        queueMicrotask(() => {
          socket.subscriber!.onFrame({ type: "pi", sessionId: "s1", record: { type: "response", id: command.id, command: "prompt", success: true } })
          socket.subscriber!.onFrame(widgetFrame("subagent-inspect", [`PI_SUBAGENT_INSPECT_JSON:${JSON.stringify({ requestId, task: "Review it", finalOutput: "LGTM" })}`]))
        })
      }
    }
    const { result } = renderHook(() => usePiSession("s1", vi.fn()))
    let reply: Awaited<ReturnType<typeof result.current.inspectSubagent>> | undefined
    await act(async () => { reply = await result.current.inspectSubagent({ asyncId: "run1", childId: "step1" }) })
    expect(reply).toMatchObject({ task: "Review it", finalOutput: "LGTM" })
    expect(result.current.widgets).toEqual({})
  })

  it("fails an inspect request when not connected", async () => {
    socket.connected = false
    const { result } = renderHook(() => usePiSession("s1", vi.fn()))
    let reply: Awaited<ReturnType<typeof result.current.inspectSubagent>> | undefined
    await act(async () => { reply = await result.current.inspectSubagent({ asyncId: "run1" }) })
    expect(reply?.error?.message).toBe("Not connected")
  })
})
