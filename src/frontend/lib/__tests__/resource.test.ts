import { describe, expect, it, vi } from "vitest"
import type { ChangeEvent } from "@/lib/backend/syncStream"
import { createResource, refresh, resetResources, resourceFamily, revalidateStale } from "@/lib/resource"

const feed = vi.hoisted(() => ({ listeners: [] as ((event: ChangeEvent) => void)[], listProjects: vi.fn() }))
vi.mock("@/lib/backend/api", async (importOriginal) => ({ ...await importOriginal<object>(), listProjects: feed.listProjects }))
vi.mock("@/lib/backend/syncStream", () => ({
  subscribeToChanges: (listener: (event: ChangeEvent) => void) => {
    feed.listeners.push(listener)
    return () => {}
  },
}))

/** A read whose requests the test settles by hand. */
function manualRead<T>() {
  const requests: { signal: AbortSignal; resolve: (value: T) => void; reject: (cause: unknown) => void }[] = []
  const read = vi.fn((signal: AbortSignal) => {
    const { promise, resolve, reject } = Promise.withResolvers<T>()
    requests.push({ signal, resolve, reject })
    return promise
  })
  return { read, requests }
}

describe("createResource", () => {
  it("shares one request between concurrent loads and stores its result", async () => {
    const { read, requests } = manualRead<number>()
    const resource = createResource(read)

    const first = resource.load()
    const second = resource.load()
    expect(read).toHaveBeenCalledOnce()
    requests[0].resolve(7)

    await expect(Promise.all([first, second])).resolves.toEqual([7, 7])
    expect(resource.get()).toBe(7)
  })

  it("keeps the request for the others when one caller aborts and cancels it once every caller did", async () => {
    const { read, requests } = manualRead<number>()
    const resource = createResource(read)
    const leaving = new AbortController()
    const staying = new AbortController()

    const left = resource.load(leaving.signal)
    const stayed = resource.load(staying.signal)
    leaving.abort(new Error("left"))
    await expect(left).rejects.toThrow("left")
    expect(requests[0].signal.aborted).toBe(false)
    requests[0].resolve(3)
    await expect(stayed).resolves.toBe(3)

    const alone = new AbortController()
    const abandoned = resource.load(alone.signal)
    alone.abort(new Error("gone"))
    await expect(abandoned).rejects.toThrow("gone")
    expect(requests[1].signal.aborted).toBe(true)
    expect(resource.get()).toBe(3)
  })

  it("re-reads on invalidate while subscribed and drops the value without subscribers", async () => {
    let reads = 0
    const resource = createResource(async () => ++reads)
    await resource.load()
    const listener = vi.fn()
    const unsubscribe = resource.subscribe(listener)

    resource.invalidate()
    expect(resource.get()).toBe(1)
    await vi.waitFor(() => expect(resource.get()).toBe(2))
    expect(listener).toHaveBeenCalledOnce()

    unsubscribe()
    resource.invalidate()
    expect(resource.get()).toBeUndefined()
    expect(reads).toBe(2)
  })

  it("notifies on set and lets only a read begun after it replace the stored value", async () => {
    const { read, requests } = manualRead<number>()
    const resource = createResource(read)
    const listener = vi.fn()
    resource.subscribe(listener)

    const earlier = resource.load()
    await Promise.resolve()
    resource.set(5)
    expect(listener).toHaveBeenCalledOnce()
    expect(resource.get()).toBe(5)

    requests[0].resolve(1)
    await vi.waitFor(() => expect(requests).toHaveLength(2))
    expect(requests[0].signal.aborted).toBe(false)
    expect(resource.get()).toBe(5)
    requests[1].resolve(6)
    await expect(earlier).resolves.toBe(6)
    expect(resource.get()).toBe(6)
    expect(listener).toHaveBeenCalledTimes(2)
  })

  it("reads once more for any number of invalidations during a read and gives every caller that answer", async () => {
    const { read, requests } = manualRead<number>()
    const resource = createResource(read)
    const first = resource.load()
    const second = resource.load()

    // One invalidation per task, as file events arrive: before the read was aborted and sent again each time (6 reads).
    for (let count = 0; count < 5; count += 1) {
      await Promise.resolve()
      resource.invalidate()
    }
    expect(read).toHaveBeenCalledOnce()
    expect(requests[0].signal.aborted).toBe(false)
    requests[0].resolve(1)
    await vi.waitFor(() => expect(requests).toHaveLength(2))
    requests[1].resolve(2)

    await expect(Promise.all([first, second])).resolves.toEqual([2, 2])
    expect(resource.get()).toBe(2)
    expect(read).toHaveBeenCalledTimes(2)
  })

  it("lets only an answer read after the latest invalidate or set become the value", async () => {
    const { requests, read } = manualRead<number>()
    const resource = createResource(read)
    const loaded = resource.load()
    requests[0].resolve(1)
    await loaded
    resource.subscribe(() => {})
    resource.invalidate()
    await Promise.resolve()

    resource.invalidate()
    requests[1].resolve(2)
    await vi.waitFor(() => expect(requests).toHaveLength(3))
    expect(resource.get()).toBe(1)
    await Promise.resolve()
    resource.set(3)
    requests[2].resolve(4)
    await vi.waitFor(() => expect(requests).toHaveLength(4))
    expect(resource.get()).toBe(3)
    requests[3].resolve(5)
    await vi.waitFor(() => expect(resource.get()).toBe(5))
  })

  it("aborts and sends again an outdated request older than 15 s, it may hang on a dead connection", async () => {
    vi.useFakeTimers({ toFake: ["Date"] })
    try {
      const { read, requests } = manualRead<number>()
      const resource = createResource(read)
      const first = resource.load()
      await Promise.resolve()

      vi.setSystemTime(Date.now() + 15_001)
      resource.invalidate()
      expect(requests[0].signal.aborted).toBe(true)
      requests[1].resolve(2)
      await expect(first).resolves.toBe(2)
    } finally {
      vi.useRealTimers()
    }
  })

  it("rejects the callers of a superseded read when its follow-up fails, not when it fails itself", async () => {
    const { read, requests } = manualRead<number>()
    const resource = createResource(read)
    const first = resource.load()
    await Promise.resolve()

    resource.invalidate()
    requests[0].reject(new Error("dropped"))
    await vi.waitFor(() => expect(requests).toHaveLength(2))
    requests[1].reject(new Error("offline"))
    await expect(first).rejects.toThrow("offline")
    expect(resource.get()).toBeUndefined()
  })

  it("keeps a revalidation running through a set and ends with its answer", async () => {
    const { read, requests } = manualRead<string[]>()
    const resource = createResource(read)
    const loaded = resource.load()
    requests[0].resolve(["a", "b"])
    await loaded
    resource.subscribe(() => {})
    resource.invalidate()
    await Promise.resolve()

    resource.set(["a"])
    expect(resource.get()).toEqual(["a"])
    expect(requests).toHaveLength(2)
    expect(requests[1].signal.aborted).toBe(false)
    const revalidated = resource.load()
    requests[1].resolve(["a", "b"])
    await vi.waitFor(() => expect(requests).toHaveLength(3))
    expect(resource.get()).toEqual(["a"])
    requests[2].resolve(["a", "c"])
    await expect(revalidated).resolves.toEqual(["a", "c"])
    expect(resource.get()).toEqual(["a", "c"])
  })

  it("reads nothing on invalidate for a value nobody loaded, even while subscribed", () => {
    const read = vi.fn(async () => 1)
    const resource = createResource(read)
    resource.subscribe(() => {})

    resource.invalidate()
    expect(read).not.toHaveBeenCalled()
  })

  it("keeps its value and notifies nobody when a re-read answers the same, by its `same`", async () => {
    const { read, requests } = manualRead<{ names: string[] }>()
    const resource = createResource(read, { same: (a, b) => JSON.stringify(a) === JSON.stringify(b) })
    const listener = vi.fn()
    resource.subscribe(listener)
    const first = { names: ["a"] }
    const loaded = resource.load()
    requests[0].resolve(first)
    await loaded
    listener.mockClear()

    resource.invalidate()
    requests[1].resolve({ names: ["a"] })
    await resource.load()
    expect(resource.get()).toBe(first)
    expect(listener).not.toHaveBeenCalled()

    resource.invalidate()
    requests[2].resolve({ names: ["b"] })
    await resource.load()
    expect(resource.get()).toEqual({ names: ["b"] })
    expect(listener).toHaveBeenCalledOnce()
  })
})

describe("resourceFamily", () => {
  it("keeps one resource per key that reads with its key", async () => {
    const read = vi.fn(async (key: string) => key.length)
    const lengths = resourceFamily(read)

    expect(lengths("abc")).toBe(lengths("abc"))
    await expect(lengths("abc").load()).resolves.toBe(3)
    expect(read).toHaveBeenCalledWith("abc", expect.any(AbortSignal))
    expect(lengths("ab").get()).toBeUndefined()
  })

  it("lists the members it created", () => {
    const lengths = resourceFamily(async (key) => key.length)
    expect(lengths.members()).toEqual([])

    const first = lengths("a")
    const second = lengths("bc")
    lengths("a")
    expect(lengths.members()).toEqual([first, second])
  })
})

describe("refresh", () => {
  it("resolves once a subscribed value is read again and keeps it when the read fails", async () => {
    const { read, requests } = manualRead<number>()
    const resource = createResource(read)
    const loaded = resource.load()
    requests[0].resolve(1)
    await loaded
    resource.subscribe(() => {})

    const refreshed = refresh(resource)
    expect(read).toHaveBeenCalledTimes(2)
    expect(resource.get()).toBe(1)
    requests[1].resolve(2)
    await refreshed
    expect(resource.get()).toBe(2)

    read.mockRejectedValueOnce(new Error("offline"))
    await refresh(resource)
    expect(resource.get()).toBe(2)
  })

  it("drops an unsubscribed value without reading", async () => {
    const read = vi.fn(async () => 1)
    const resource = createResource(read)
    await resource.load()

    await refresh(resource)
    expect(resource.get()).toBeUndefined()
    expect(read).toHaveBeenCalledOnce()
  })
})

describe("revalidateStale", () => {
  it("reads a subscribed value again after its re-read failed, until one succeeds", async () => {
    const read = vi.fn<() => Promise<number>>().mockResolvedValueOnce(1).mockRejectedValueOnce(new Error("offline")).mockResolvedValueOnce(2)
    const resource = createResource(read)
    await resource.load()
    resource.subscribe(() => {})
    await refresh(resource)
    expect(resource.get()).toBe(1)

    revalidateStale()
    await vi.waitFor(() => expect(resource.get()).toBe(2))
    revalidateStale()
    expect(read).toHaveBeenCalledTimes(3)
  })
})

describe("resetResources", () => {
  it("forgets every value, family members included", async () => {
    const single = createResource(async () => 1)
    const family = resourceFamily(async (key) => key)
    single.set(2)
    await family("kept").load()

    resetResources()
    expect(single.get()).toBeUndefined()
    expect(family("kept").get()).toBeUndefined()
  })
})

describe("projectsResource", () => {
  it("reads the projects again when the project catalog changes", async () => {
    const { projectsResource, watchDocumentLists } = await import("@/lib/backend/resources")
    feed.listProjects.mockResolvedValueOnce([]).mockResolvedValueOnce([{ slug: "p", name: "P", tabs: [] }])
    const stop = watchDocumentLists()
    await projectsResource.load()
    projectsResource.subscribe(() => {})

    for (const listener of feed.listeners) {
      listener({ seq: 1, resource_kind: "document", resource_key: "chat/projects-meta.json", version: null, operation: "write", device_id: null })
    }
    await vi.waitFor(() => expect(projectsResource.get()).toEqual([{ slug: "p", name: "P", tabs: [] }]))
    stop()
  })
})
