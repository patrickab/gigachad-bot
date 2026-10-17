"use client"

import { useCallback, useState } from "react"

const COPIED_MS = 1500

/** Copies `text` to the clipboard and reports `copied` for a moment afterwards. */
export function useCopyFeedback(text: string): { copied: boolean; copy: () => void } {
  const [copied, setCopied] = useState(false)
  const copy = useCallback(() => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), COPIED_MS)
    })
  }, [text])
  return { copied, copy }
}
