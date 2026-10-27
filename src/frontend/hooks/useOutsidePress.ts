"use client"

import type { RefObject } from "react"
import { useEventListener } from "@/hooks/useEventListener"

/** Call `onOutside` on a document press outside `ref` while `active`; no Escape handling. */
export function useOutsidePress(
  ref: RefObject<HTMLElement | null>,
  onOutside: () => void,
  { active = true, event = "mousedown", capture = false }: { active?: boolean; event?: "mousedown" | "pointerdown"; capture?: boolean } = {},
) {
  useEventListener("document", event, (e) => {
    if (ref.current && !ref.current.contains(e.target as Node)) onOutside()
  }, { active, capture })
}
