"use client"

import { useEffect, useEffectEvent, type RefObject } from "react"
import { useOutsidePress } from "@/hooks/useOutsidePress"

/** Call `handler` on a mousedown outside `ref` or on Escape. */
export function useClickOutside(ref: RefObject<HTMLElement | null>, handler: () => void) {
  useOutsidePress(ref, handler)
  const onKeyDown = useEffectEvent((e: KeyboardEvent) => {
    if (e.key === "Escape") handler()
  })

  useEffect(() => {
    const listener = (e: KeyboardEvent) => onKeyDown(e)
    document.addEventListener("keydown", listener)
    return () => document.removeEventListener("keydown", listener)
  }, [])
}
