"use client"

import { useCallback, useEffect, useState } from "react"
import { Check, Copy } from "lucide-react"
import { highlightCode } from "@/lib/markdown-syntax-highlighting"

/** Syntax-highlighted code with a hover-revealed copy button. Shared by markdown-rendered
 *  fenced code and any other surface that needs to show a script (e.g. a tool call's source). */
export function CodeBlock({ codeString, language }: { codeString: string; language: string }) {
  const [copied, setCopied] = useState(false)
  const [html, setHtml] = useState("")

  useEffect(() => {
    let cancelled = false
    highlightCode(codeString, language).then((result) => {
      if (!cancelled) setHtml(result)
    })
    return () => { cancelled = true }
  }, [codeString, language])

  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(codeString).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    })
  }, [codeString])

  return (
    <div className="group/code relative">
      {html ? (
        <div
          className="[&_pre]:!rounded-md [&_pre]:!p-4 [&_pre]:!m-0 [&_pre]:!text-[0.75rem] [&_pre]:!leading-[1.6]"
          dangerouslySetInnerHTML={{ __html: html }}
        />
      ) : (
        <pre className="rounded-md p-4 m-0 text-[0.75rem] leading-[1.6] bg-surface text-ink overflow-x-auto"><code>{codeString}</code></pre>
      )}
      <button
        onClick={handleCopy}
        className="absolute top-2 right-2 rounded p-1 opacity-0 transition-opacity group-hover/code:opacity-100 hover:bg-surface-elevated/60 text-ink-subtle hover:text-ink"
      >
        {copied ? <Check className="h-3.5 w-3.5 text-ink" /> : <Copy className="h-3.5 w-3.5" />}
      </button>
    </div>
  )
}
