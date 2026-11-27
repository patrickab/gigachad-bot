import { cleanup, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { useFollowFiles } from "@/hooks/useFollowFiles"
import type { ChangeEvent } from "@/lib/backend/syncStream"

vi.mock("@/lib/backend/urls", () => ({ fileWatchUrl: () => "ws://backend/files/watch" }))

// The sync feed of change events: what a stored key hears.
const feed = vi.hoisted(() => ({ listeners: new Set<(event: ChangeEvent) => void>() }))
vi.mock("@/lib/backend/syncStream", () => ({
  subscribeToChanges: (listener: (event: ChangeEvent) => void) => {
    feed.listeners.add(listener)
    return () => { feed.listeners.delete(listener) }
  },
}))

class FakeSocket {
  static readonly CONNECTING = 0
  static readonly OPEN = 1
  static readonly CLOSED = 3
  static instances: FakeSocket[] = []
  readyState = FakeSocket.CONNECTING
  closedByClient = false
  sent: string[] = []
  onopen: (() => void) | null = null
  onmessage: ((event: { data: unknown }) => void) | null = null
  onclose: (() => void) | null = null

  constructor(readonly url: string) {
    FakeSocket.instances.push(this)
  }

  send(data: string) {
    if (this.readyState !== FakeSocket.OPEN) throw new Error("send on a socket that is not open")
    this.sent.push(data)
  }

  close() {
    this.closedByClient = true
    this.readyState = FakeSocket.CLOSED
    this.onclose?.()
  }

  open() {
    this.readyState = FakeSocket.OPEN
    this.onopen?.()
  }

  drop() {
    this.readyState = FakeSocket.CLOSED
    this.onclose?.()
  }

  receive(message: object) { this.onmessage?.({ data: JSON.stringify(message) }) }

  get frames(): Record<string, unknown>[] { return this.sent.map((frame) => JSON.parse(frame)) }
}

const SETTLED_MS = 100

/** Let the debounced set update go out and open the socket it created. */
function connect(): FakeSocket {
  vi.advanceTimersByTime(SETTLED_MS)
  const socket = FakeSocket.instances.at(-1)!
  socket.open()
  return socket
}

const changed = (path: string): ChangeEvent => ({ seq: 1, resource_kind: "document", resource_key: path, version: null, operation: "write", device_id: "other" })

beforeEach(() => {
  vi.useFakeTimers()
  vi.stubGlobal("WebSocket", FakeSocket)
})

afterEach(() => {
  cleanup()
  feed.listeners.clear()
  vi.runOnlyPendingTimers()
  FakeSocket.instances = []
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

describe("useFollowFiles watching", () => {
  it("watches absolute paths only: a stored key is not watched but still hears the sync feed", () => {
    const onChange = vi.fn()
    renderHook(() => useFollowFiles(["project/notes.md", "/abs/a.tex"], onChange, ["project", "/abs/dir"]))
    const socket = connect()
    expect(socket.frames).toEqual([{ type: "watch", files: ["/abs/a.tex"], folders: ["/abs/dir"] }])

    for (const listener of feed.listeners) listener(changed("project/notes.md"))
    for (const listener of feed.listeners) listener(changed("project/chapters/two.md"))
    expect(onChange.mock.calls).toEqual([["project/notes.md"], ["project/chapters/two.md"]])
  })
})

describe("useFollowFiles changes", () => {
  it("reports a change of a followed file and of any file inside a followed folder, nothing else", () => {
    const onChange = vi.fn()
    renderHook(() => useFollowFiles(["/abs/a.tex"], onChange, ["/abs/dir"]))
    const socket = connect()

    socket.receive({ type: "changed", path: "/abs/a.tex" })
    socket.receive({ type: "changed", path: "/abs/dir/deep/b.tex" })
    socket.receive({ type: "changed", path: "/abs/dirx/c.tex" })
    socket.receive({ type: "changed", path: "/abs/other.tex" })

    expect(onChange.mock.calls).toEqual([["/abs/a.tex"], ["/abs/dir/deep/b.tex"]])
  })
})
