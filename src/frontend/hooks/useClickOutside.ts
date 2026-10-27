"use client"

import type { RefObject } from "react"
import { useEventListener } from "@/hooks/useEventListener"
import { useOutsidePress } from "@/hooks/useOutsidePress"

/** Call `handler` on a mousedown outside `ref` or on Escape. */
export function useClickOutside(ref: RefObject<HTMLElement | null>, handler: () => void) {
  useOutsidePress(ref, handler)
  useEventListener("document", "keydown", (e) => {
    if (e.key === "Escape") handler()
  })
}
