import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { onFileChange } from "../fileEvents"
import { watchFiles, type WatchTargets } from "../fileWatch"

vi.mock("../urls", () => ({ fileWatchUrl: () => "ws://backend/files/watch" }))
vi.mock("../syncStream", () => ({ subscribeToChanges: () => () => {} }))

// What happened in order: the watch frames the sockets sent and the paths the listeners heard.
const timeline: string[] = []

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
    timeline.push("watch frame")
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

  /** The connection breaks under the client. */
  drop() {
    this.readyState = FakeSocket.CLOSED
    this.onclose?.()
  }

  receive(message: object) { this.onmessage?.({ data: JSON.stringify(message) }) }

  receiveRaw(data: unknown) { this.onmessage?.({ data }) }

  get frames(): Record<string, unknown>[] { return this.sent.map((frame) => JSON.parse(frame)) }
}

const SETTLED_MS = 100
const cleanups: Array<() => void> = []

/** Register `targets` like an editor does, and release it after the test. */
function follow({ files = [], folders = [] }: Partial<WatchTargets>): () => void {
  const release = watchFiles({ files, folders })
  cleanups.push(release)
  return release
}

/** Listen to file changes like an editor does, and stop after the test. */
function listen(match: string | ((path: string) => boolean), listener: (path: string) => void): void {
  cleanups.push(onFileChange(match, listener))
}

/** Let the debounced set update go out and open the socket it created. */
function connect(): FakeSocket {
  vi.advanceTimersByTime(SETTLED_MS)
  const socket = FakeSocket.instances.at(-1)!
  socket.open()
  return socket
}

beforeEach(() => {
  vi.useFakeTimers()
  vi.stubGlobal("WebSocket", FakeSocket)
})

afterEach(() => {
  for (const cleanup of cleanups.splice(0)) cleanup()
  vi.runOnlyPendingTimers()
  FakeSocket.instances = []
  timeline.length = 0
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

describe("watchFiles registrations", () => {
  it("sends a path held by two registrations once and unwatches it only after both left", () => {
    const first = follow({ files: ["/a.tex"] })
    const second = follow({ files: ["/a.tex", "/b.tex"], folders: ["/proj"] })
    follow({ files: ["/c.tex"] })
    const socket = connect()
    expect(socket.frames).toEqual([{ type: "watch", files: ["/a.tex", "/b.tex", "/c.tex"], folders: ["/proj"] }])

    first()
    vi.advanceTimersByTime(SETTLED_MS)
    expect(socket.frames).toHaveLength(1)

    second()
    vi.advanceTimersByTime(SETTLED_MS)
    expect(socket.frames.at(-1)).toEqual({ type: "watch", files: ["/c.tex"], folders: [] })
  })

  it("releases a registration once however often it is called, and counts a path listed twice once", () => {
    const first = follow({ files: ["/a.tex", "/a.tex"] })
    const second = follow({ files: ["/a.tex"] })
    const socket = connect()

    first()
    first()
    vi.advanceTimersByTime(SETTLED_MS)
    expect(socket.frames).toHaveLength(1)
    expect(socket.closedByClient).toBe(false)

    second()
    vi.advanceTimersByTime(SETTLED_MS)
    expect(socket.closedByClient).toBe(true)
  })

  it("opens one socket for the first registration, closes it after the last left and does not reconnect", () => {
    expect(FakeSocket.instances).toHaveLength(0)
    const first = follow({ files: ["/a.tex"] })
    const second = follow({ folders: ["/proj"] })
    const socket = connect()
    expect(FakeSocket.instances).toHaveLength(1)
    expect(socket.url).toBe("ws://backend/files/watch")

    first()
    vi.advanceTimersByTime(SETTLED_MS)
    expect(socket.closedByClient).toBe(false)

    second()
    vi.advanceTimersByTime(SETTLED_MS)
    expect(socket.closedByClient).toBe(true)
    vi.advanceTimersByTime(60_000)
    expect(FakeSocket.instances).toHaveLength(1)
  })
})

describe("watchFiles set updates", () => {
  it("debounces registrations into one watch frame and sends nothing for an unchanged set", () => {
    follow({ files: ["/a.tex"] })
    vi.advanceTimersByTime(10)
    follow({ files: ["/b.tex"] })
    vi.advanceTimersByTime(10)
    expect(FakeSocket.instances).toHaveLength(0)

    const socket = connect()
    expect(socket.frames).toEqual([{ type: "watch", files: ["/a.tex", "/b.tex"], folders: [] }])

    follow({ files: ["/c.tex"] })()
    vi.advanceTimersByTime(SETTLED_MS)
    expect(socket.frames).toHaveLength(1)

    follow({ files: ["/d.tex"] })
    vi.advanceTimersByTime(10)
    follow({ folders: ["/proj"] })
    vi.advanceTimersByTime(SETTLED_MS)
    expect(socket.frames).toHaveLength(2)
    expect(socket.frames[1]).toEqual({ type: "watch", files: ["/a.tex", "/b.tex", "/d.tex"], folders: ["/proj"] })
  })

  it("sends the whole set once when a socket that is still connecting opens", () => {
    follow({ files: ["/a.tex"] })
    vi.advanceTimersByTime(SETTLED_MS)
    const socket = FakeSocket.instances[0]!
    follow({ files: ["/b.tex"] })
    vi.advanceTimersByTime(SETTLED_MS)
    expect(socket.sent).toEqual([])

    socket.open()
    expect(socket.frames).toEqual([{ type: "watch", files: ["/a.tex", "/b.tex"], folders: [] }])
  })
})

describe("watchFiles changes", () => {
  it("reports a changed file to every onFileChange listener, a folder's inner file included", () => {
    const heard = vi.fn()
    const exact = vi.fn()
    listen(() => true, heard)
    listen("/a.tex", exact)
    follow({ files: ["/a.tex"], folders: ["/proj"] })
    const socket = connect()

    socket.receive({ type: "changed", path: "/a.tex" })
    socket.receive({ type: "changed", path: "/proj/chapters/intro.tex" })

    expect(heard.mock.calls).toEqual([["/a.tex"], ["/proj/chapters/intro.tex"]])
    expect(exact.mock.calls).toEqual([["/a.tex"]])
  })

  it("ignores frames that are not a change of a path", () => {
    const heard = vi.fn()
    listen(() => true, heard)
    follow({ files: ["/a.tex"] })
    const socket = connect()

    socket.receive({ type: "refused", path: "/a.tex", message: "outside the roots" })
    socket.receive({ type: "changed" })
    socket.receive({ type: "changed", path: 7 })
    socket.receiveRaw("not json")
    socket.receiveRaw(new ArrayBuffer(4))

    expect(heard).not.toHaveBeenCalled()
  })
})

describe("watchFiles reconnect", () => {
  it("re-sends the set, with what was registered meanwhile, and reports every watched path once after the frame", () => {
    listen(() => true, (path) => timeline.push(`changed ${path}`))
    follow({ files: ["/a.tex", "/b.tex"], folders: ["/proj"] })
    const first = connect()
    expect(timeline).toEqual(["watch frame"])

    first.drop()
    follow({ files: ["/c.tex"] })
    vi.advanceTimersByTime(999)
    expect(FakeSocket.instances).toHaveLength(1)
    vi.advanceTimersByTime(1)
    expect(FakeSocket.instances).toHaveLength(2)

    const second = FakeSocket.instances[1]!
    second.open()
    expect(second.frames).toEqual([{ type: "watch", files: ["/a.tex", "/b.tex", "/c.tex"], folders: ["/proj"] }])
    expect(timeline).toEqual(["watch frame", "watch frame", "changed /a.tex", "changed /b.tex", "changed /c.tex", "changed /proj"])

    vi.advanceTimersByTime(SETTLED_MS)
    expect(second.frames).toHaveLength(1)
    expect(timeline).toHaveLength(6)
  })

  it("waits longer after every failed attempt and starts over once a socket opened", () => {
    follow({ files: ["/a.tex"] })
    vi.advanceTimersByTime(SETTLED_MS)
    FakeSocket.instances[0]!.drop()
    vi.advanceTimersByTime(999)
    expect(FakeSocket.instances).toHaveLength(1)
    vi.advanceTimersByTime(1)
    expect(FakeSocket.instances).toHaveLength(2)

    FakeSocket.instances[1]!.drop()
    vi.advanceTimersByTime(1999)
    expect(FakeSocket.instances).toHaveLength(2)
    vi.advanceTimersByTime(1)
    expect(FakeSocket.instances).toHaveLength(3)

    FakeSocket.instances[2]!.open()
    FakeSocket.instances[2]!.drop()
    vi.advanceTimersByTime(999)
    expect(FakeSocket.instances).toHaveLength(3)
    vi.advanceTimersByTime(1)
    expect(FakeSocket.instances).toHaveLength(4)
  })

  it("does not reconnect when the last registration left during the outage", () => {
    const release = follow({ files: ["/a.tex"] })
    const socket = connect()

    socket.drop()
    release()
    vi.advanceTimersByTime(60_000)

    expect(FakeSocket.instances).toHaveLength(1)
  })
})
