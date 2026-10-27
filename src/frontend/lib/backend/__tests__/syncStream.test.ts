import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

class FakeEventSource {
  static instances: FakeEventSource[] = []
  readonly handlers = new Map<string, (message: MessageEvent<string>) => void>()
  onerror: (() => void) | null = null
  closed = false

  constructor(readonly url: string) {
    FakeEventSource.instances.push(this)
  }

  addEventListener(name: string, handler: (message: MessageEvent<string>) => void): void {
    this.handlers.set(name, handler)
  }

  emit(name: string, data: unknown): void {
    this.handlers.get(name)?.({ data: JSON.stringify(data) } as MessageEvent<string>)
  }

  close(): void {
    this.closed = true
  }
}

const change = (seq: number, resource_key = "notes/a.md") =>
  ({ seq, resource_kind: "document", resource_key, version: 1, device_id: "other-device" })

async function load() {
  vi.resetModules()
  return import("@/lib/backend/syncStream")
}

const settled = async (promise: Promise<void>) => {
  let done = false
  void promise.then(() => { done = true })
  await Promise.resolve()
  await Promise.resolve()
  return done
}

beforeEach(() => {
  FakeEventSource.instances = []
  vi.useFakeTimers()
  vi.stubGlobal("EventSource", FakeEventSource)
})

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

describe("sync stream cursor", () => {
  it("starts a fresh page at the announced head and reconnects from the last applied change", async () => {
    const { subscribeToChanges, syncCursorReady } = await load()
    const ready = syncCursorReady()
    const received: number[] = []
    subscribeToChanges((event) => received.push(event.seq))

    const [first] = FakeEventSource.instances
    expect(FakeEventSource.instances).toHaveLength(1)
    expect(first.url).toMatch(/\/sync\/stream$/)
    expect(await settled(ready)).toBe(false)

    first.emit("cursor", { seq: 41 })
    expect(await settled(ready)).toBe(true)
    first.emit("change", change(42))
    expect(received).toEqual([42])

    first.onerror?.()
    vi.advanceTimersByTime(1000)
    expect(FakeEventSource.instances[1].url).toMatch(/\/sync\/stream\?since=42$/)
  })

  it("replays from the start after a first connection that never announced the head", async () => {
    const { subscribeToChanges, syncCursorReady } = await load()
    subscribeToChanges(() => {})
    const ready = syncCursorReady()

    FakeEventSource.instances[0].onerror?.()
    expect(await settled(ready)).toBe(true)
    vi.advanceTimersByTime(1000)
    expect(FakeEventSource.instances[1].url).toMatch(/\/sync\/stream\?since=0$/)
  })

  it("lets reads start at the first replayed change from a server that sends no head", async () => {
    const { subscribeToChanges, syncCursorReady } = await load()
    const received: number[] = []
    subscribeToChanges((event) => received.push(event.seq))
    const ready = syncCursorReady()

    FakeEventSource.instances[0].emit("change", change(3))
    expect(await settled(ready)).toBe(true)
    expect(received).toEqual([3])
    vi.advanceTimersByTime(5_000)
    expect(FakeEventSource.instances[0].closed).toBe(false)
  })

  it("stops waiting for a head that does not arrive in time", async () => {
    const { subscribeToChanges, syncCursorReady } = await load()
    subscribeToChanges(() => {})
    const ready = syncCursorReady()

    vi.advanceTimersByTime(5_000)
    expect(await settled(ready)).toBe(true)
    expect(FakeEventSource.instances[0].closed).toBe(true)
    vi.advanceTimersByTime(1000)
    expect(FakeEventSource.instances[1].url).toMatch(/\/sync\/stream\?since=0$/)
  })

  it("keeps a stream that owes the head open until it arrives, then closes it when unused", async () => {
    const { subscribeToChanges, syncCursorReady } = await load()
    const unsubscribe = subscribeToChanges(() => {})
    const ready = syncCursorReady()
    unsubscribe()

    const [stream] = FakeEventSource.instances
    expect(stream.closed).toBe(false)
    stream.emit("cursor", { seq: 7 })
    expect(await settled(ready)).toBe(true)
    expect(stream.closed).toBe(true)

    subscribeToChanges(() => {})
    expect(FakeEventSource.instances[1].url).toMatch(/\/sync\/stream\?since=7$/)
  })
})
