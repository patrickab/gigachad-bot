// Shared store for backend values every mounted tab reads, e.g. the model catalog.
// Consumes one read function per value. Provides a cached value per resource that every subscriber sees, one request
// however many callers load or invalidate in one task, direct writes for mutation responses, families of keyed resources
// that list their members, an awaitable re-read, a retry of every re-read that failed, and a reset that forgets every value.

/** One backend value shared by every subscriber. */
export interface Resource<T> {
  /** Latest value, undefined until the first load resolved. */
  get: () => T | undefined
  subscribe: (listener: () => void) => () => void
  /** Read from the backend. Concurrent callers share one request, one caller's abort never cancels it for the others. */
  load: (signal?: AbortSignal) => Promise<T>
  /**
   * Read a subscribed value again, drop an unsubscribed one so the next subscriber loads, and read nothing for a value never loaded.
   * A read in progress starts over and its callers get the new answer. A re-read that fails is retried by `revalidateStale`.
   */
  invalidate: () => void
  /** Store `value`, e.g. a mutation's response. A read in progress starts over, so only an answer read after `value` replaces it. */
  set: (value: T) => void
}

/** One read in progress, shared by every `load` caller that joined it, restarted by every `invalidate` and `set`. */
interface Flight<T> {
  /** Settles with the answer of the request current when one arrives. */
  promise: Promise<T>
  resolve: (value: T) => void
  reject: (cause: unknown) => void
  /** Aborts the current request. */
  controller: AbortController
  /** The current request began in this task, so it already sees every change announced so far. */
  fresh: boolean
  /** Callers still waiting, the request aborts once every one of them aborted. */
  waiting: number
  /** A re-read of a subscribed value, which runs to its end however many callers leave. */
  held: boolean
}

const resets = new Set<() => void>()
/** The retry of every resource whose latest invalidation has no successful read yet. */
const staleRetries = new Set<() => void>()
const ignore = () => {}

/** Share `read`'s result between every caller and subscriber. */
export function createResource<T>(read: (signal: AbortSignal) => Promise<T>): Resource<T> {
  let value: T | undefined
  // The read whose answer becomes the value. Only its current request counts, an answer begun before a restart is dropped.
  let flight: Flight<T> | null = null
  const listeners = new Set<() => void>()

  /** Send `current`'s request, whose answer settles it unless a restart or the last caller's leaving replaced it. */
  const request = (current: Flight<T>) => {
    const { controller } = current
    current.fresh = true
    queueMicrotask(() => { current.fresh = false })
    read(controller.signal).then(
      (result) => {
        if (flight !== current || current.controller !== controller) return
        flight = null
        value = result
        staleRetries.delete(retry)
        listeners.forEach((listener) => listener())
        current.resolve(result)
      },
      (cause: unknown) => {
        if (flight !== current || current.controller !== controller) return
        flight = null
        current.reject(cause)
      },
    )
  }

  const begin = (): Flight<T> => {
    const { promise, resolve, reject } = Promise.withResolvers<T>()
    // Every caller observes the outcome through its own promise, so a read all of them left rejects unobserved.
    promise.catch(ignore)
    const current: Flight<T> = { promise, resolve, reject, controller: new AbortController(), fresh: false, waiting: 0, held: false }
    flight = current
    request(current)
    return current
  }

  /** Abort `current`'s request and send it again, unless it began in this task and so is not outdated. */
  const restart = (current: Flight<T>) => {
    if (current.fresh) return
    current.controller.abort()
    current.controller = new AbortController()
    request(current)
  }

  const load = (signal?: AbortSignal): Promise<T> => {
    if (signal?.aborted) return Promise.reject(signal.reason)
    const current = flight ?? begin()
    current.waiting += 1
    const { promise, resolve, reject } = Promise.withResolvers<T>()
    const leave = () => {
      reject(signal?.reason)
      current.waiting -= 1
      if (current.waiting > 0 || current.held) return
      // Nobody is left to use the answer.
      current.controller.abort()
      if (flight === current) flight = null
    }
    signal?.addEventListener("abort", leave, { once: true })
    void current.promise.then(resolve, reject).finally(() => signal?.removeEventListener("abort", leave))
    return promise
  }

  const invalidate = () => {
    if (flight) restart(flight)
    if (listeners.size === 0) {
      // Nobody shows the value: forget it, the next subscriber reads it.
      value = undefined
      staleRetries.delete(retry)
      return
    }
    // A value nobody loaded stays unloaded until its first reader asks.
    if (value === undefined) return
    const current = flight ?? begin()
    current.held = true
    staleRetries.add(retry)
  }

  function retry() {
    if (!flight) invalidate()
  }

  resets.add(() => {
    flight = null
    value = undefined
  })

  return {
    get: () => value,
    subscribe: (listener) => {
      listeners.add(listener)
      return () => { listeners.delete(listener) }
    },
    load,
    invalidate,
    set: (next) => {
      value = next
      if (flight) restart(flight)
      listeners.forEach((listener) => listener())
    },
  }
}

/** One resource per key, created on first use, and the members created so far. */
export interface ResourceFamily<T> {
  (key: string): Resource<T>
  /** Every member created so far, e.g. to drop a deleted file from each list. */
  members: () => Resource<T>[]
}

/** One resource per key, created on first use, e.g. the reasoning levels of each model. */
export function resourceFamily<T>(read: (key: string, signal: AbortSignal) => Promise<T>): ResourceFamily<T> {
  const members = new Map<string, Resource<T>>()
  const member = (key: string) => {
    let found = members.get(key)
    if (!found) {
      found = createResource((signal) => read(key, signal))
      members.set(key, found)
    }
    return found
  }
  return Object.assign(member, { members: () => [...members.values()] })
}

/** `invalidate` `resource`, resolving once a value subscribers see is read again. A failed read keeps that value. */
export async function refresh<T>(resource: Resource<T>): Promise<void> {
  resource.invalidate()
  // Only a subscribed value survives `invalidate`, and its read is already running.
  if (resource.get() !== undefined) await resource.load().catch(ignore)
}

/** Invalidate every resource whose latest re-read failed and that no read is running for. */
export function revalidateStale(): void {
  [...staleRetries].forEach((retry) => retry())
}

/** Forget every resource's value and read in progress without notifying, so one test's values never reach the next. */
export function resetResources(): void {
  staleRetries.clear()
  resets.forEach((reset) => reset())
}
