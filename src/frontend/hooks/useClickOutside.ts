"use client"

import { useEffect, type RefObject } from "react"

export function useClickOutside(ref: RefObject<HTMLElement | null>, handler: () => void) {
  useEffect(() => {
    function onEvent(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) handler()
    }
    document.addEventListener("mousedown", onEvent)
    return () => document.removeEventListener("mousedown", onEvent)
  }, [ref, handler])
}