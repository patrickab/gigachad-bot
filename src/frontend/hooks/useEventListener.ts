"use client"

import { useEffect, useEffectEvent } from "react"

type ListenerOptions = { active?: boolean; capture?: boolean }

/** Listen to `type` on `window` or `document` while `active`, always calling the latest `handler`. */
export function useEventListener<K extends keyof WindowEventMap>(target: "window", type: K, handler: (event: WindowEventMap[K]) => void, options?: ListenerOptions): void
export function useEventListener<K extends keyof DocumentEventMap>(target: "document", type: K, handler: (event: DocumentEventMap[K]) => void, options?: ListenerOptions): void
export function useEventListener(
  target: "window" | "document",
  type: string,
  handler: (event: Event) => void,
  { active = true, capture = false }: ListenerOptions = {},
) {
  const onEvent = useEffectEvent(handler)

  useEffect(() => {
    if (!active) return
    const eventTarget = target === "window" ? window : document
    const listener = (event: Event) => onEvent(event)
    eventTarget.addEventListener(type, listener, capture)
    return () => eventTarget.removeEventListener(type, listener, capture)
  }, [target, type, active, capture])
}
