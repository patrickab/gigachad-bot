"use client"

import { useEffect, type RefObject } from "react"

/** Call `handler` on a mousedown outside `ref` or on Escape. Memoize `handler` to avoid re-subscribing every render. */
export function useClickOutside(ref: RefObject<HTMLElement | null>, handler: () => void) {
  useEffect(() => {
    function onMouseDown(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) handler()
    }
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") handler()
    }
    document.addEventListener("mousedown", onMouseDown)
    document.addEventListener("keydown", onKeyDown)
    return () => {
      document.removeEventListener("mousedown", onMouseDown)
      document.removeEventListener("keydown", onKeyDown)
    }
  }, [ref, handler])
}