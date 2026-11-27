import { act, renderHook, waitFor } from "@testing-library/react"
import type { ReactNode } from "react"
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest"
import type { PiSessionInfo } from "@/lib/backend/api"
import { liveSession } from "../liveSession"
import { markPiSessionRead, piSessionEvents, PiSessionsProvider, searchPiSessions, usePiSessions, usePiSessionStatus, type PiSessionEvent } from "../sessionStore"

const api = vi.hoisted(() => ({
  sessions: [] as unknown[],
  fetchPiSessions: vi.fn(),
  fetchPiRepositories: vi.fn(),
  searchPiSessions: vi.fn(),
}))

vi.mock("@/lib/backend/api", () => ({
  fetchPiSessions: api.fetchPiSessions,
  fetchPiRepositories: api.fetchPiRepositories,
  searchPiSessions: api.searchPiSessions,
}))

/** A WebSocket the test opens and feeds frames into. */
class FakeSocket {
  static OPEN = 1
  static instances: FakeSocket[] = []
  readyState = 0
  sent: { type: string; sessionId?: string; command?: { id: string; type: string } }[] = []
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
}

function socket(): FakeSocket {
  const current = FakeSocket.instances.at(-1)!
  current.open()
  return current
}

function session(sessionId: string, cwd: string, running = false): PiSessionInfo {
  return {
    sessionId, sessionPath: `/sessions/${sessionId}.jsonl`, cwd, machine: "local", createdAt: "2026-01-01T00:00:00Z", modifiedAt: "2026-01-01T00:00:00Z",
    firstMessage: "hi", messageCount: 1, tokens: 0, live: running ? { running, waiting: false, attached: 0 } : null, pinned: false, archived: false,
  }
}

const wrapper = ({ children }: { children: ReactNode }) => <PiSessionsProvider>{children}</PiSessionsProvider>

const cleanups: (() => void)[] = []

/** Attach `sessionId` as a tab would, idle, and return a helper that sends its `pi` records. */
function attach(sessionId: string) {
  cleanups.push(liveSession(sessionId).attach())
  const host = socket()
  host.emit({ type: "attached", sessionId, streamId: "x", state: { isStreaming: false }, ui: { dialogs: [], statuses: {}, queue: { steering: [], followUp: [] } }, entries: { entries: [], leafId: null } })
  host.emit({ type: "replay_start", sessionId })
  host.emit({ type: "replay_end", sessionId })
  return (record: object) => host.emit({ type: "pi", sessionId, record })
}

function recordEvents(): PiSessionEvent[] {
  const events: PiSessionEvent[] = []
  cleanups.push(piSessionEvents.subscribe((event) => events.push(event)))
  return events
}

const stored = new Map<string, string>()

beforeAll(() => {
  vi.stubGlobal("WebSocket", FakeSocket)
  vi.stubGlobal("localStorage", { getItem: (key: string) => stored.get(key) ?? null, setItem: (key: string, value: string) => { stored.set(key, value) } })
})

afterEach(() => {
  cleanups.splice(0).forEach((cleanup) => cleanup())
  stored.clear()
})

api.fetchPiRepositories.mockResolvedValue([{ name: "alpha", path: "/home/u/git/alpha" }, { name: "beta", path: "/home/u/git/beta" }])
api.fetchPiSessions.mockImplementation(async () => structuredClone(api.sessions))

describe("usePiSessions", () => {
  it("lists every repo's sessions and refreshes on sessions_changed", async () => {
    api.sessions = [session("a1", "/home/u/git/alpha"), session("b1", "/home/u/git/beta"), session("a2", "/home/u/git/alpha/sub")]
    const { result, unmount } = renderHook(() => usePiSessions(), { wrapper })
    cleanups.push(unmount)
    await waitFor(() => expect(result.current.sessions.map((listed) => listed.sessionId)).toEqual(["a1", "b1", "a2"]))
    expect(api.fetchPiSessions).toHaveBeenCalledWith()

    api.sessions = [session("a3", "/home/u/git/alpha"), ...api.sessions]
    socket().emit({ type: "sessions_changed", sessionPaths: ["/sessions/a3.jsonl"] })
    await waitFor(() => expect(result.current.sessions.map((listed) => listed.sessionId)).toEqual(["a3", "a1", "b1", "a2"]))
  })

  it("lists only coding sessions for profile coding, counting a session without a profile as coding (G5)", async () => {
    api.sessions = [
      session("c1", "/home/u/git/alpha"),
      { ...session("d1", "/home/u/workbench/markdown"), profile: "document" },
      { ...session("c2", "/home/u/git/alpha"), profile: "coding" },
    ]
    const { result, unmount } = renderHook(() => ({ coding: usePiSessions({ profile: "coding" }), all: usePiSessions() }), { wrapper })
    cleanups.push(unmount)
    await waitFor(() => expect(result.current.coding.sessions.map((listed) => listed.sessionId)).toEqual(["c1", "c2"]))
    expect(result.current.all.sessions).toHaveLength(3)
  })

  it("drops search hits of other profiles", async () => {
    api.searchPiSessions.mockResolvedValue([
      { session: session("c1", "/home/u/git/alpha"), snippet: "a", entryId: "e1" },
      { session: { ...session("d1", "/home/u/workbench/pdf"), profile: "document" }, snippet: "b", entryId: "e2" },
    ])
    expect((await searchPiSessions("x", { profile: "coding" })).map((hit) => hit.session.sessionId)).toEqual(["c1"])
    expect(await searchPiSessions("x")).toHaveLength(2)
  })

  it("marks a listed run that stopped unseen as unread", async () => {
    api.sessions = [session("l1", "/home/u/git/alpha", true)]
    const { result, unmount } = renderHook(() => ({ list: usePiSessions(), status: usePiSessionStatus("l1") }), { wrapper })
    cleanups.push(unmount)
    await waitFor(() => expect(result.current.status.running).toBe(true))

    api.sessions = [session("l1", "/home/u/git/alpha")]
    socket().emit({ type: "sessions_changed", sessionPaths: [] })
    await waitFor(() => expect(result.current.status).toEqual({ running: false, waiting: false, unread: true, failed: false }))
  })
})

describe("list refresh cost", () => {
  /** Let a notice's list read, if any, finish. */
  const settle = () => act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)) })

  it("reads the list again only for a notice that can concern a listed session", async () => {
    api.sessions = [session("f1", "/home/u/git/alpha")]
    const { result, unmount } = renderHook(() => usePiSessions(), { wrapper })
    cleanups.push(unmount)
    await waitFor(() => expect(result.current.loading).toBe(false))
    await settle()
    api.fetchPiSessions.mockClear()

    for (let notice = 0; notice < 5; notice++) {
      socket().emit({ type: "sessions_changed", sessionPaths: [`/home/u/.pi/agent/factory/sessions/--repo--/${notice}.jsonl`, "/home/u/.pi/agent/factory/tasks/t.lock"] })
      await settle()
    }
    expect(api.fetchPiSessions).toHaveBeenCalledTimes(0)

    for (const sessionPaths of [["/home/u/.pi/agent/factory/sessions/a.jsonl", "/home/u/factory/agent/sessions/b.jsonl"], [], undefined]) {
      api.fetchPiSessions.mockClear()
      socket().emit({ type: "sessions_changed", sessionPaths })
      await settle()
      expect(api.fetchPiSessions).toHaveBeenCalledTimes(1)
    }
  })

  it("keeps an unchanged list and its listeners untouched, but publishes a failure and the recovery", async () => {
    api.sessions = [session("u1", "/home/u/git/alpha"), session("u2", "/home/u/git/beta", true)]
    let renders = 0
    const { result, unmount } = renderHook(() => { renders += 1; return usePiSessions() }, { wrapper })
    cleanups.push(unmount)
    await waitFor(() => expect(result.current.sessions).toHaveLength(2))
    await settle()
    const shown = result.current.sessions
    renders = 0

    for (let read = 0; read < 5; read++) {
      socket().emit({ type: "sessions_changed", sessionPaths: [] })
      await settle()
    }
    expect(api.fetchPiSessions).toHaveBeenCalled()
    expect(renders).toBe(0)
    expect(result.current.sessions).toBe(shown)

    api.fetchPiSessions.mockRejectedValueOnce(new Error("offline"))
    socket().emit({ type: "sessions_changed", sessionPaths: [] })
    await waitFor(() => expect(result.current.error).toBe("offline"))
    socket().emit({ type: "sessions_changed", sessionPaths: [] })
    await waitFor(() => expect(result.current.error).toBeNull())
    expect(result.current.sessions.map((listed) => listed.sessionId)).toEqual(["u1", "u2"])
  })
})

describe("usePiSessionStatus", () => {
  it("goes running, settled and unread, then read, with settled emitted once and not on agent_end", () => {
    const { result, unmount } = renderHook(() => usePiSessionStatus("s1"), { wrapper })
    cleanups.push(unmount)
    const events = recordEvents()
    const send = attach("s1")

    send({ type: "agent_start" })
    expect(result.current).toEqual({ running: true, waiting: false, unread: false, failed: false })
    send({ type: "agent_end", messages: [{ role: "assistant", stopReason: "stop" }], willRetry: false })
    expect(result.current.running).toBe(true)
    expect(events).toEqual([])

    send({ type: "agent_settled" })
    send({ type: "agent_settled" })
    expect(result.current).toEqual({ running: false, waiting: false, unread: true, failed: false })
    expect(events).toEqual([{ type: "settled", sessionId: "s1", aborted: false }])
    expect(JSON.parse(localStorage.getItem("pi-unread-sessions")!)).toContain("s1")

    act(() => markPiSessionRead("s1"))
    expect(result.current.unread).toBe(false)
  })

  it("reports an aborted run and a failed one", () => {
    cleanups.push(renderHook(() => usePiSessionStatus("s2"), { wrapper }).unmount)
    const events = recordEvents()
    const send = attach("s2")
    send({ type: "agent_start" })
    send({ type: "agent_end", messages: [{ role: "assistant", stopReason: "aborted" }], willRetry: false })
    send({ type: "agent_settled" })
    send({ type: "agent_start" })
    send({ type: "agent_end", messages: [{ role: "assistant", stopReason: "error" }], willRetry: false })
    send({ type: "agent_settled" })
    expect(events).toEqual([
      { type: "settled", sessionId: "s2", aborted: true },
      { type: "failed", sessionId: "s2" },
      { type: "settled", sessionId: "s2", aborted: false },
    ])
    act(() => markPiSessionRead("s2"))
  })

  it("waits on a dialog until ui_closed", () => {
    const { result, unmount } = renderHook(() => usePiSessionStatus("s3"), { wrapper })
    cleanups.push(unmount)
    const events = recordEvents()
    const send = attach("s3")
    send({ type: "extension_ui_request", id: "d1", method: "confirm", title: "Proceed?" })
    expect(result.current.waiting).toBe(true)
    expect(events).toEqual([{ type: "waiting", sessionId: "s3" }])

    socket().emit({ type: "ui_closed", sessionId: "s3", id: "d1" })
    expect(result.current.waiting).toBe(false)
  })

  it("shows context usage and cost from get_session_stats while attached", async () => {
    const { result, unmount } = renderHook(() => usePiSessionStatus("s4"), { wrapper })
    cleanups.push(unmount)
    const send = attach("s4")
    const request = socket().sent.find((frame) => frame.sessionId === "s4" && frame.command?.type === "get_session_stats")!
    send({ type: "response", id: request.command!.id, command: "get_session_stats", success: true, data: { cost: 0.45, contextUsage: { percent: 30 } } })
    await waitFor(() => expect(result.current).toMatchObject({ contextPercent: 30, cost: 0.45 }))

    act(() => cleanups.pop()!())
    expect(result.current.contextPercent).toBeUndefined()
  })

  it("asks get_session_stats once on attach, after an answer and after the run settles (measure)", () => {
    cleanups.push(renderHook(() => usePiSessionStatus("s5"), { wrapper }).unmount)
    const asked = () => socket().sent.filter((frame) => frame.sessionId === "s5" && frame.command?.type === "get_session_stats").length
    const send = attach("s5")
    expect(asked()).toBe(1)
    send({ type: "agent_start" })
    send({ type: "message_end", message: { role: "assistant", content: [], stopReason: "stop", timestamp: 1 } })
    expect(asked()).toBe(2)
    send({ type: "agent_settled" })
    // Was 5: the session store asked on its own beside the view (2 + 1 + 2)
    expect(asked()).toBe(3)
    act(() => markPiSessionRead("s5"))
  })
})
