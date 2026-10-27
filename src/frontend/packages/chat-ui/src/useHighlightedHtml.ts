"use client"

import { useEffect, useState } from "react"
import { highlightCode } from "./syntaxHighlighting.ts"

/** Highlighted HTML of `code`, "" until the async highlight resolves (and while not `enabled`); superseded results are dropped. */
export function useHighlightedHtml(code: string, lang: string, enabled = true): string {
  const [html, setHtml] = useState("")

  useEffect(() => {
    if (!enabled) return
    let cancelled = false
    highlightCode(code, lang).then((result) => {
      if (!cancelled) setHtml(result)
    }).catch(() => {
      if (!cancelled) setHtml("")
    })
    return () => { cancelled = true }
  }, [code, lang, enabled])

  return html
}
