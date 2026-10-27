"use client"

import { useLayoutEffect, useState, type RefObject } from "react"

type Size = { width: number; height: number }

const ZERO_SIZE: Size = { width: 0, height: 0 }

/** Rounded rendered size of the element mounted in `ref`, measured before paint and tracked with a `ResizeObserver`; `initial` until the first measurement. */
export function useElementSize(ref: RefObject<HTMLElement | null>, initial: Size = ZERO_SIZE): Size {
  const [size, setSize] = useState(initial)
  useLayoutEffect(() => {
    const element = ref.current
    if (!element) return
    const update = (width: number, height: number) => {
      const next = { width: Math.round(width), height: Math.round(height) }
      setSize((current) => current.width === next.width && current.height === next.height ? current : next)
    }
    update(element.offsetWidth, element.offsetHeight)
    if (typeof ResizeObserver === "undefined") return
    const observer = new ResizeObserver(([entry]) => {
      if (entry) update(entry.contentRect.width, entry.contentRect.height)
    })
    observer.observe(element)
    return () => observer.disconnect()
  }, [ref])
  return size
}
