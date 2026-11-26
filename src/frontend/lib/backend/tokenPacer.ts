/** Jitter buffer for streamed text: chunks go in as they arrive, `emit` gets them at a steady pace.
 *
 *  Each frame drains `backlog / LAG_MS` characters (backlog smoothed by an exponential average), so the
 *  queue settles where the drain speed equals the arrival speed and the display trails by about `LAG_MS`.
 *  A late chunk is covered by the cushion, a burst is absorbed within the lag. */
const LAG_MS = 200
const FINISH_LAG_MS = 60
const MIN_CHARS_PER_SECOND = 30
const BACKLOG_SMOOTHING = 0.2
const MAX_FRAME_MS = 100
/** Least time between two emits: a 120 or 144 Hz display would else render the reply twice as often as it reads any smoother. */
const MIN_FRAME_MS = 12

export interface TokenPacer {
  push(text: string): void
  /** Emit everything queued now (abort, error, or text that must precede the next event). */
  flush(): void
  /** Drain the queue quickly, then resolve. */
  finish(): Promise<void>
}

interface Scheduler {
  request(callback: () => void): number
  cancel(handle: number): void
}

/** Frames stop in a hidden or unfocused window (Firefox, occluded windows), and the text must still arrive: a timer takes over after this. */
const FRAME_FALLBACK_MS = 100

let nextHandle = 1
const pendingFrames = new Map<number, { frame: number; timer: ReturnType<typeof setTimeout> }>()

const browserScheduler: Scheduler = {
  request: (callback) => {
    const handle = nextHandle++
    const run = () => {
      const pending = pendingFrames.get(handle)
      if (!pending) return
      pendingFrames.delete(handle)
      cancelAnimationFrame(pending.frame)
      clearTimeout(pending.timer)
      callback()
    }
    pendingFrames.set(handle, { frame: requestAnimationFrame(run), timer: setTimeout(run, FRAME_FALLBACK_MS) })
    return handle
  },
  cancel: (handle) => {
    const pending = pendingFrames.get(handle)
    if (!pending) return
    pendingFrames.delete(handle)
    cancelAnimationFrame(pending.frame)
    clearTimeout(pending.timer)
  },
}

export function createTokenPacer(emit: (text: string) => void, scheduler: Scheduler = browserScheduler): TokenPacer {
  let queue = ""
  let handle: number | null = null
  let last = 0
  let credit = 0
  let smoothed = 0
  let lag = LAG_MS
  let waiters: (() => void)[] = []

  const settle = () => {
    const done = waiters
    waiters = []
    for (const resolve of done) resolve()
  }

  const stop = () => {
    if (handle !== null) scheduler.cancel(handle)
    handle = null
  }

  const frame = () => {
    handle = null
    // Not the rAF argument: its origin differs from `performance.now()` in some environments.
    const time = performance.now()
    if (time - last < MIN_FRAME_MS && lag === LAG_MS) {
      handle = scheduler.request(frame)
      return
    }
    const dt = Math.min(time - last, MAX_FRAME_MS)
    last = time
    smoothed += (queue.length - smoothed) * BACKLOG_SMOOTHING
    const rate = Math.max(MIN_CHARS_PER_SECOND, (smoothed * 1000) / lag)
    credit += (rate * dt) / 1000
    let count = Math.floor(credit)
    credit -= count
    if (count > 0) {
      // Never split a surrogate pair.
      const code = queue.charCodeAt(count - 1)
      if (code >= 0xd800 && code <= 0xdbff && count < queue.length) count += 1
      emit(queue.slice(0, count))
      queue = queue.slice(count)
    }
    if (queue.length > 0) {
      handle = scheduler.request(frame)
    } else {
      credit = 0
      smoothed = 0
      settle()
    }
  }

  const start = () => {
    if (handle !== null) return
    last = performance.now()
    // A reply starts with its first character, not after a delay
    if (smoothed === 0 && credit === 0) credit = 1
    handle = scheduler.request(frame)
  }

  return {
    push(text) {
      if (!text) return
      queue += text
      start()
    },
    flush() {
      stop()
      if (queue) emit(queue)
      queue = ""
      credit = 0
      smoothed = 0
      settle()
    },
    finish() {
      if (!queue) return Promise.resolve()
      lag = FINISH_LAG_MS
      return new Promise<void>((resolve) => {
        waiters.push(resolve)
        start()
      })
    },
  }
}
