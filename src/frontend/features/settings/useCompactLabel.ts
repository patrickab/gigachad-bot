"use client"

import { useEffect, useRef, useState } from "react"

/** Whether the expanded trigger label would cover `chatColumn`, measured on every resize of the trigger, label, column and main area. */
export function useCompactLabel(chatColumn: HTMLElement | null | undefined, loading: boolean) {
  const [compact, setCompact] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)
  const labelRef = useRef<HTMLSpanElement>(null)
  const contentRef = useRef<HTMLSpanElement>(null)

  useEffect(() => {
    const root = rootRef.current
    const label = labelRef.current
    const content = contentRef.current
    const main = root?.closest("main")
    if (!chatColumn || !root || !label || !content || !main) return
    const measure = () => {
      // Keep overlap detection stable throughout the label's width animation.
      const expandedRight = root.getBoundingClientRect().right + content.getBoundingClientRect().width - label.getBoundingClientRect().width
      setCompact(expandedRight > chatColumn.getBoundingClientRect().left)
    }
    const observer = new ResizeObserver(measure)
    for (const element of [root, content, chatColumn, main]) observer.observe(element)
    measure()
    return () => observer.disconnect()
  }, [chatColumn, loading])

  return { compact: Boolean(chatColumn && compact), rootRef, labelRef, contentRef }
}
