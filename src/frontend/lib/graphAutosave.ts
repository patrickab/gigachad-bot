"use client"

import { useCallback, useEffect, useRef, useState } from "react"

export const AUTOSAVE_DEBOUNCE_MS = 650
// A pure debounce never fires while typing continues; this bounds the wait.
export const AUTOSAVE_MAX_WAIT_MS = 5000

interface Handlers {
  write: (content: string) => Promise<unknown>
  onSaved?: (content: string) => void
  onError?: (cause: unknown) => void
}

interface Pending {
  content: string
  handlers: Handlers
}

// One view per graph file may hold pending work. A background view hands off by
// flushing immediately, so it can never wake on a timer and clobber the new owner.
const owners = new Map<string, { token: object; flush: () => void }>()

export interface GraphAutosave {
  /** Debounced write. Ignored when content matches what was last written. */
  queue: (content: string) => void
  /** Write any pending content now. */
  flush: () => void
  /** Drop pending content without writing it (invalid input). */
  cancel: () => void
  /** Set the baseline written content, e.g. straight after a load. */
  markSaved: (content: string | null) => void
  /** True while unsaved content exists. */
  dirty: boolean
}

interface Options extends Handlers {
  /** Graph file this view writes to; ownership is tracked per key. */
  key: string
  debounceMs?: number
  maxWaitMs?: number
}

export function useGraphAutosave({
  key,
  write,
  onSaved,
  onError,
  debounceMs = AUTOSAVE_DEBOUNCE_MS,
  maxWaitMs = AUTOSAVE_MAX_WAIT_MS,
}: Options): GraphAutosave {
  const [dirty, setDirty] = useState(false)
  const token = useRef({}).current
  const saved = useRef<string | null>(null)
  const pending = useRef<Pending | null>(null)
  const debounceTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const maxWaitTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const chain = useRef(Promise.resolve())
  // Latest handlers, captured into each pending write so a key change mid-flight
  // cannot send the previous file's content to the next file.
  const handlers = useRef<Handlers>({ write, onSaved, onError })
  handlers.current = { write, onSaved, onError }

  const clearTimers = useCallback(() => {
    if (debounceTimer.current) clearTimeout(debounceTimer.current)
    if (maxWaitTimer.current) clearTimeout(maxWaitTimer.current)
    debounceTimer.current = null
    maxWaitTimer.current = null
  }, [])

  const flush = useCallback(() => {
    clearTimers()
    const next = pending.current
    pending.current = null
    if (!next || next.content === saved.current) {
      setDirty(false)
      return
    }
    const run = chain.current.then(async () => {
      await next.handlers.write(next.content)
      saved.current = next.content
      setDirty(false)
      next.handlers.onSaved?.(next.content)
    })
    // Serialise writes so a slow one cannot land after a newer one.
    chain.current = run.catch((cause: unknown) => next.handlers.onError?.(cause))
  }, [clearTimers])

  const flushRef = useRef(flush)
  flushRef.current = flush

  const queue = useCallback((content: string) => {
    if (content === saved.current) {
      pending.current = null
      clearTimers()
      setDirty(false)
      return
    }
    const owner = owners.get(key)
    if (owner && owner.token !== token) owner.flush()
    owners.set(key, { token, flush: () => flushRef.current() })
    pending.current = { content, handlers: handlers.current }
    setDirty(true)
    if (debounceTimer.current) clearTimeout(debounceTimer.current)
    debounceTimer.current = setTimeout(() => flushRef.current(), debounceMs)
    // Armed once per burst and never reset, so sustained typing still writes.
    if (!maxWaitTimer.current) maxWaitTimer.current = setTimeout(() => flushRef.current(), maxWaitMs)
  }, [key, token, clearTimers, debounceMs, maxWaitMs])

  const cancel = useCallback(() => {
    clearTimers()
    pending.current = null
  }, [clearTimers])

  const markSaved = useCallback((content: string | null) => {
    clearTimers()
    pending.current = null
    saved.current = content
    setDirty(false)
  }, [clearTimers])

  useEffect(() => {
    const onVisibility = () => {
      if (document.visibilityState === "hidden") flushRef.current()
    }
    document.addEventListener("visibilitychange", onVisibility)
    return () => document.removeEventListener("visibilitychange", onVisibility)
  }, [])

  useEffect(() => () => {
    flushRef.current()
    if (owners.get(key)?.token === token) owners.delete(key)
  }, [key, token])

  return { queue, flush, cancel, markSaved, dirty }
}
