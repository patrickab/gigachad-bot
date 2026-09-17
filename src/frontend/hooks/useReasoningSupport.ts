"use client"

import { useEffect, useRef, useState } from "react"
import { fetchReasoningSupport } from "@/lib/api"

/** Whether `model` accepts `reasoning_effort`, per LiteLLM's model metadata.
 *  `undefined` while the lookup is in flight (or not yet started) so callers
 *  can avoid acting on a not-yet-known result. Caches per model string for
 *  the component's lifetime; resolves to `false` on lookup failure. */
export function useReasoningSupport(model: string): boolean | undefined {
  const cache = useRef(new Map<string, boolean>())
  const [supported, setSupported] = useState<boolean | undefined>(() => cache.current.get(model))

  useEffect(() => {
    const cached = cache.current.get(model)
    if (cached !== undefined) {
      setSupported(cached)
      return
    }
    setSupported(undefined)
    let cancelled = false
    fetchReasoningSupport(model)
      .then(({ supports_reasoning }) => {
        cache.current.set(model, supports_reasoning)
        if (!cancelled) setSupported(supports_reasoning)
      })
      .catch(() => {
        if (!cancelled) setSupported(false)
      })
    return () => {
      cancelled = true
    }
  }, [model])

  return supported
}
