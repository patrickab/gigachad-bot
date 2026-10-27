import { useEffect, useState, type RefObject } from "react"
import { readStorage, writeStorage } from "@/lib/storage"

export const LAYOUTS = ["unified", "split", "auto"] as const
type DiffLayout = (typeof LAYOUTS)[number]

const LAYOUT_STORAGE_KEY = "diff-layout"
const SPLIT_MIN_WIDTH = 1100

function storedLayout(): DiffLayout {
  return LAYOUTS.find((layout) => layout === readStorage(LAYOUT_STORAGE_KEY)) ?? "auto"
}

/** Persist layout choice and resolve auto from pane width. */
export function useDiffLayout(container: RefObject<HTMLElement | null>) {
  const [layout, setLayout] = useState(storedLayout)
  const [wide, setWide] = useState(false)
  useEffect(() => {
    const element = container.current
    if (!element || typeof ResizeObserver === "undefined") return
    const observer = new ResizeObserver(([entry]) => setWide(entry!.contentRect.width >= SPLIT_MIN_WIDTH))
    observer.observe(element)
    return () => observer.disconnect()
  }, [container])
  const choose = (next: DiffLayout) => {
    setLayout(next)
    writeStorage(LAYOUT_STORAGE_KEY, next)
  }
  return { layout, choose, split: layout === "split" || (layout === "auto" && wide) }
}
