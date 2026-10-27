"use client"

import { useEffect, useEffectEvent, type RefObject } from "react"

/** Call `onOutside` on a document press outside `ref` while `active`; no Escape handling. */
export function useOutsidePress(
  ref: RefObject<HTMLElement | null>,
  onOutside: () => void,
  { active = true, event = "mousedown", capture = false }: { active?: boolean; event?: "mousedown" | "pointerdown"; capture?: boolean } = {},
) {
  const onPress = useEffectEvent((e: Event) => {
    if (ref.current && !ref.current.contains(e.target as Node)) onOutside()
  })

  useEffect(() => {
    if (!active) return
    const listener = (e: Event) => onPress(e)
    document.addEventListener(event, listener, capture)
    return () => document.removeEventListener(event, listener, capture)
  }, [active, event, capture])
}
