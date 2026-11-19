import { act, renderHook, waitFor } from "@testing-library/react"
import type { ReactNode } from "react"
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest"
import type { PiSessionInfo } from "@/lib/backend/api"
import { subscribePiSession } from "../piSocket"
import { markPiSessionRead, piSessionEvents, PiSessionsProvider, usePiSessions, usePiSessionStatus, type PiSessionEvent } from "../sessionStore"

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
    firstMessage: "hi", messageCount: 1, live: running ? { running, waiting: false, attached: 0 } : null, pinned: false, archived: false,
  }
}

const wrapper = ({ children }: { children: ReactNode }) => <PiSessionsProvider>{children}</PiSessionsProvider>

const cleanups: (() => void)[] = []

/** Attach `sessionId` as a tab would, idle, and return a helper that sends its `pi` records. */
function attach(sessionId: string) {
  cleanups.push(subscribePiSession(sessionId, { onFrame: () => {}, onStatus: () => {} }))
  const host = socket()
  host.emit({ type: "attached", sessionId, state: { isStreaming: false }, ui: { dialogs: [], statuses: {}, queue: { steering: [], followUp: [] } }, entries: [] })
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
api.fetchPiSessions.mockImplementation(async () => api.sessions)

describe("usePiSessions", () => {
  it("filters every repo's sessions by repo and refreshes on sessions_changed", async () => {
    api.sessions = [session("a1", "/home/u/git/alpha"), session("b1", "/home/u/git/beta"), session("a2", "/home/u/git/alpha/sub")]
    const { result, unmount } = renderHook(() => usePiSessions({ repo: "alpha" }), { wrapper })
    cleanups.push(unmount)
    await waitFor(() => expect(result.current.sessions.map((listed) => listed.sessionId)).toEqual(["a1", "a2"]))
    expect(api.fetchPiSessions).toHaveBeenCalledWith(undefined, { archived: false })

    api.sessions = [session("a3", "/home/u/git/alpha"), ...api.sessions]
    socket().emit({ type: "sessions_changed", sessionPaths: ["/sessions/a3.jsonl"] })
    await waitFor(() => expect(result.current.sessions.map((listed) => listed.sessionId)).toEqual(["a3", "a1", "a2"]))
  })

  it("marks a listed run that stopped unseen as unread", async () => {
    api.sessions = [session("l1", "/home/u/git/alpha", true)]
    const { result, unmount } = renderHook(() => ({ list: usePiSessions(), status: usePiSessionStatus("l1") }), { wrapper })
    cleanups.push(unmount)
    await waitFor(() => expect(result.current.status.running).toBe(true))

    api.sessions = [session("l1", "/home/u/git/alpha")]
    socket().emit({ type: "sessions_changed", sessionPaths: [] })
    await waitFor(() => expect(result.current.status).toEqual({ running: false, waiting: false, unread: true }))
  })
})

describe("usePiSessionStatus", () => {
  it("goes running, settled and unread, then read, with settled emitted once and not on agent_end", () => {
    const { result, unmount } = renderHook(() => usePiSessionStatus("s1"), { wrapper })
    cleanups.push(unmount)
    const events = recordEvents()
    const send = attach("s1")

    send({ type: "agent_start" })
    expect(result.current).toEqual({ running: true, waiting: false, unread: false })
    send({ type: "agent_end", messages: [{ role: "assistant", stopReason: "stop" }], willRetry: false })
    expect(result.current.running).toBe(true)
    expect(events).toEqual([])

    send({ type: "agent_settled" })
    send({ type: "agent_settled" })
    expect(result.current).toEqual({ running: false, waiting: false, unread: true })
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

  it("shows context usage and cost from get_session_stats while attached", () => {
    const { result, unmount } = renderHook(() => usePiSessionStatus("s4"), { wrapper })
    cleanups.push(unmount)
    const send = attach("s4")
    const request = socket().sent.find((frame) => frame.sessionId === "s4" && frame.command?.type === "get_session_stats")!
    send({ type: "response", id: request.command!.id, command: "get_session_stats", success: true, data: { cost: 0.45, contextUsage: { percent: 30 } } })
    expect(result.current).toMatchObject({ contextPercent: 30, cost: 0.45 })

    act(() => cleanups.pop()!())
    expect(result.current.contextPercent).toBeUndefined()
  })
})
