import { describe, expect, it } from "vitest"
import type { HostFrame } from "../piSocket"
import { createResyncGuard, SessionResync } from "../resync"

const SESSION = "session-1"

function entry(id: string, parentId: string | null) {
  return { type: "message", id, parentId, timestamp: "2026-01-01T00:00:00.000Z", message: { role: "user", content: id, timestamp: 0 } }
}

function frame(type: string, streamId: string | null, fields: Record<string, unknown> = {}): HostFrame {
  return { type, sessionId: SESSION, streamId, sequence: 0, ...fields }
}

function attached(streamId: string, leafId: string | null) {
  return frame("attached", streamId, { state: {}, ui: {}, entries: { entries: [], leafId } })
}

function resyncAnswer(streamId: string, leafId: string | null, entries: unknown[] = []) {
  return frame("resync", streamId, { mode: "append", entries, leafId })
}

/** A tracker attached on stream `s1` at cursor `a`, recording what it sends. */
function attachedTracker(online = true) {
  const sent: HostFrame[] = []
  const sync = new SessionResync(SESSION, (sentFrame) => {
    if (online) sent.push(sentFrame)
    return online
  })
  sync.receive(attached("s1", "a"))
  return { sync, sent }
}

describe("createResyncGuard", () => {
  it("holds one request, reports one follow-up and ignores stale tokens", () => {
    const guard = createResyncGuard()
    const first = guard.tryAcquire()!
    expect(guard.tryAcquire()).toBeNull()
    expect(guard.tryAcquire()).toBeNull()
    expect(guard.release(first)).toBe(true)
    const second = guard.tryAcquire()!
    expect(guard.release(second)).toBe(false)

    const old = guard.tryAcquire()!
    guard.reset()
    const fresh = guard.tryAcquire()!
    expect(guard.owns(old)).toBe(false)
    expect(guard.release(old)).toBe(false)
    expect(guard.owns(fresh)).toBe(true)
  })
})

describe("SessionResync", () => {
  it("does nothing before the session is attached", () => {
    const sent: HostFrame[] = []
    const sync = new SessionResync(SESSION, (sentFrame) => sent.push(sentFrame) > 0)
    sync.request()
    expect(sent).toEqual([])
  })

  it("collapses a burst into one request in flight plus one follow-up from the new cursor", () => {
    const { sync, sent } = attachedTracker()
    for (let trigger = 0; trigger < 5; trigger++) sync.request()
    expect(sent).toEqual([{ type: "resync", sessionId: SESSION, cursor: "a" }])

    expect(sync.receive(resyncAnswer("s1", "b", [entry("b", "a")]))).toEqual({ streamChanged: false, apply: true })
    expect(sent.slice(1)).toEqual([{ type: "resync", sessionId: SESSION, cursor: "b" }])

    expect(sync.receive(resyncAnswer("s1", "b")).apply).toBe(true)
    expect(sent).toHaveLength(2)
  })

  it("on a new streamId drops the request in flight and resyncs from cursor null", () => {
    const { sync, sent } = attachedTracker()
    sync.request()
    expect(sync.receive(frame("pi", "s2", { record: { type: "agent_start" } }))).toEqual({ streamChanged: true, apply: true })
    expect(sent.slice(1)).toEqual([{ type: "resync", sessionId: SESSION, cursor: null }])

    // The answer to the request from before the change is stale, the one after it applies
    expect(sync.receive(resyncAnswer("s2", "x")).apply).toBe(false)
    expect(sync.receive(resyncAnswer("s2", "y")).apply).toBe(true)
    sync.request()
    expect(sent.at(-1)).toEqual({ type: "resync", sessionId: SESSION, cursor: "y" })
  })

  it("drops a resync answer stamped with no stream it follows, freeing the guard", () => {
    const { sync, sent } = attachedTracker()
    sync.request()
    expect(sync.receive(frame("resync", null, { mode: "append", entries: [], leafId: "z" })).apply).toBe(false)
    sync.request()
    expect(sent).toEqual([{ type: "resync", sessionId: SESSION, cursor: "a" }, { type: "resync", sessionId: SESSION, cursor: "a" }])
  })

  it("treats a new stream on the resync answer itself as a reset", () => {
    const { sync, sent } = attachedTracker()
    sync.request()
    expect(sync.receive(resyncAnswer("s2", "q"))).toEqual({ streamChanged: true, apply: false })
    expect(sent.at(-1)).toEqual({ type: "resync", sessionId: SESSION, cursor: null })
  })

  it("resyncs on reconnect when a stream is known, attaches when none is", () => {
    const { sync, sent } = attachedTracker()
    sync.request()
    sync.disconnected()
    sync.reconnected()
    expect(sent.at(-1)).toEqual({ type: "resync", sessionId: SESSION, cursor: "a" })

    const fresh: HostFrame[] = []
    const unattached = new SessionResync(SESSION, (sentFrame) => fresh.push(sentFrame) > 0)
    unattached.reconnected()
    expect(fresh).toEqual([{ type: "attach", sessionId: SESSION }])
  })

  it("falls back to a full reattach when the host does not know resync", () => {
    const { sync, sent } = attachedTracker()
    sync.request()
    sync.request()
    const refused = frame("error", "s1", { message: "unknown frame type \"resync\"" })
    expect(sync.receive(refused).apply).toBe(false)
    expect(sent.slice(1)).toEqual([{ type: "detach", sessionId: SESSION }, { type: "attach", sessionId: SESSION }])

    sync.request()
    expect(sent).toHaveLength(3)
    sync.receive(attached("s1", "b"))
    sync.request()
    expect(sent.slice(3)).toEqual([{ type: "detach", sessionId: SESSION }, { type: "attach", sessionId: SESSION }])

    // Answer the follow-up and its queued trigger; then a live message_end waits for the next reattach instead of forcing one
    sync.receive(attached("s1", "c"))
    sync.receive(attached("s1", "d"))
    expect(sent).toHaveLength(7)
    sync.request("message_end")
    expect(sent).toHaveLength(7)
  })

  it("passes other errors through and frees the guard when the socket is down", () => {
    const { sync } = attachedTracker()
    expect(sync.receive(frame("error", "s1", { message: "prompt refused" })).apply).toBe(true)

    const offline = attachedTracker(false)
    offline.sync.request()
    offline.sync.request()
    expect(offline.sent).toEqual([])
  })
})
