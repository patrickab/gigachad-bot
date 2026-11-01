import { cleanup, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { useFollowFiles } from "@/hooks/useFollowFiles"
import { announceFileChange } from "@/lib/backend/fileEvents"
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

  it("opens no socket when it only follows stored keys", () => {
    renderHook(() => useFollowFiles(["project/notes.md"], vi.fn(), ["project"]))
    vi.advanceTimersByTime(SETTLED_MS)

    expect(FakeSocket.instances).toHaveLength(0)
  })

  it("keeps its registration across renders with fresh arrays, changes it with the content and calls the latest onChange", () => {
    const first = vi.fn()
    const second = vi.fn()
    const view = renderHook(
      ({ files, onChange }) => useFollowFiles(files, onChange, ["/abs/dir"]),
      { initialProps: { files: ["/abs/a.tex", "/abs/b.tex"], onChange: first } },
    )
    const socket = connect()

    view.rerender({ files: ["/abs/b.tex", "/abs/a.tex", "/abs/a.tex"], onChange: second })
    vi.advanceTimersByTime(SETTLED_MS)
    expect(socket.frames).toHaveLength(1)
    expect(FakeSocket.instances).toHaveLength(1)

    socket.receive({ type: "changed", path: "/abs/a.tex" })
    expect(first).not.toHaveBeenCalled()
    expect(second.mock.calls).toEqual([["/abs/a.tex"]])

    view.rerender({ files: ["/abs/a.tex"], onChange: second })
    vi.advanceTimersByTime(SETTLED_MS)
    expect(socket.frames.at(-1)).toEqual({ type: "watch", files: ["/abs/a.tex"], folders: ["/abs/dir"] })
    socket.receive({ type: "changed", path: "/abs/b.tex" })
    expect(second).toHaveBeenCalledTimes(1)
  })

  it("stops following and unwatches when it unmounts", () => {
    const onChange = vi.fn()
    const view = renderHook(() => useFollowFiles(["/abs/a.tex"], onChange))
    const socket = connect()

    view.unmount()
    vi.advanceTimersByTime(SETTLED_MS)
    announceFileChange("/abs/a.tex")

    expect(socket.closedByClient).toBe(true)
    expect(onChange).not.toHaveBeenCalled()
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

  it("also reports a change this device announced", () => {
    const onChange = vi.fn()
    renderHook(() => useFollowFiles(["/abs/a.tex"], onChange, ["/abs/dir"]))
    connect()

    announceFileChange("/abs/a.tex")
    announceFileChange("/abs/dir/b.tex")
    announceFileChange("/abs/other.tex")

    expect(onChange.mock.calls).toEqual([["/abs/a.tex"], ["/abs/dir/b.tex"]])
  })

  it("reports every followed file and folder once after a reconnect", () => {
    const onChange = vi.fn()
    renderHook(() => useFollowFiles(["/abs/a.tex", "/abs/b.tex"], onChange, ["/abs/dir"]))
    connect().drop()
    expect(onChange).not.toHaveBeenCalled()

    vi.advanceTimersByTime(1000)
    const second = FakeSocket.instances[1]!
    second.open()

    expect(second.frames).toEqual([{ type: "watch", files: ["/abs/a.tex", "/abs/b.tex"], folders: ["/abs/dir"] }])
    expect(onChange.mock.calls).toEqual([["/abs/a.tex"], ["/abs/b.tex"], ["/abs/dir"]])
  })
})
