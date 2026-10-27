"use client"

import { useEffect, useEffectEvent, type RefObject } from "react"

type ListenerOptions = { active?: boolean; capture?: boolean; passive?: boolean }

/** Listen to `type` on `window`, `document` or the element in a ref while `active`, always calling the latest `handler`. A ref target subscribes to the element mounted when the effect runs. */
export function useEventListener<K extends keyof WindowEventMap>(target: "window", type: K, handler: (event: WindowEventMap[K]) => void, options?: ListenerOptions): void
export function useEventListener<K extends keyof DocumentEventMap>(target: "document", type: K, handler: (event: DocumentEventMap[K]) => void, options?: ListenerOptions): void
export function useEventListener<K extends keyof HTMLElementEventMap>(target: RefObject<HTMLElement | null>, type: K, handler: (event: HTMLElementEventMap[K]) => void, options?: ListenerOptions): void
export function useEventListener(
  target: "window" | "document" | RefObject<HTMLElement | null>,
  type: string,
  handler: (event: Event) => void,
  { active = true, capture = false, passive }: ListenerOptions = {},
) {
  const onEvent = useEffectEvent(handler)

  useEffect(() => {
    if (!active) return
    const eventTarget = target === "window" ? window : target === "document" ? document : target.current
    if (!eventTarget) return
    const listener = (event: Event) => onEvent(event)
    eventTarget.addEventListener(type, listener, { capture, passive })
    return () => eventTarget.removeEventListener(type, listener, capture)
  }, [target, type, active, capture, passive])
}
