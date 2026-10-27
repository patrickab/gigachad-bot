"use client"

import { useCallback, useEffect, useRef, useState } from "react"

const COPIED_MS = 1500

/** Copies `text` to the clipboard and reports `copied` for a moment afterwards. */
export function useCopyFeedback(text: string): { copied: boolean; copy: () => void } {
  const [copied, setCopied] = useState(false)
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  useEffect(() => () => clearTimeout(timer.current), [])
  const copy = useCallback(() => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true)
      clearTimeout(timer.current)
      timer.current = setTimeout(() => setCopied(false), COPIED_MS)
    })
  }, [text])
  return { copied, copy }
}
