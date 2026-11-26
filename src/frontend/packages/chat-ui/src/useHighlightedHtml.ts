"use client"

import { useEffect, useMemo, useState } from "react"
import { createIncrementalHighlighter, highlightCode, type HighlightedLines } from "./syntaxHighlighting.ts"

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

/**
 * Highlighted lines of `code`, null until the async highlight resolves (and while not `enabled`).
 * For code that grows at its end: a streamed fence tokenizes each line once, not the whole block per character.
 */
export function useHighlightedLines(code: string, lang: string, enabled = true): HighlightedLines | null {
  const highlighter = useMemo(() => createIncrementalHighlighter(lang), [lang])
  const [result, setResult] = useState<{ highlighter: unknown; lines: HighlightedLines | null } | null>(null)

  useEffect(() => {
    if (!enabled) return
    let cancelled = false
    highlighter.highlight(code).then((lines) => {
      if (!cancelled) setResult({ highlighter, lines })
    }).catch(() => {
      if (!cancelled) setResult(null)
    })
    return () => { cancelled = true }
  }, [code, highlighter, enabled])

  // Lines of another language's highlighter are never shown
  return enabled && result?.highlighter === highlighter ? result.lines : null
}
