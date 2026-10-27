"use client"

import { Check, Copy } from "lucide-react"
import { useHighlightedHtml } from "@/components/content/useHighlightedHtml"
import { useCopyFeedback } from "@/hooks/useCopyFeedback"
import { cn } from "@/lib/utils"

/** Syntax-highlighted code with a hover-revealed copy button. Shared by markdown-rendered
 *  fenced code and any other surface that needs to show a script (e.g. a tool call's source).
 *  `plain` skips highlighting, for diagrams drawn with text characters. */
export function CodeBlock({ codeString, language, plain = false }: { codeString: string; language: string; plain?: boolean }) {
  const { copied, copy } = useCopyFeedback(codeString)
  // Highlighting is async, so the plain `<pre>` shows until it resolves.
  const html = useHighlightedHtml(codeString, language, !plain)

  return (
    <div className={cn("group/code relative", plain && "my-3")} data-code-language={language === "text" ? "" : language}>
      {html ? (
        // The `!` overrides beat the inline styles the highlighter puts on its `<pre>`.
        <div
          className="[&_pre]:!rounded-md [&_pre]:!p-4 [&_pre]:!m-0 [&_pre]:!text-[0.75rem] [&_pre]:!leading-[1.6]"
          dangerouslySetInnerHTML={{ __html: html }}
        />
      ) : (
        <pre className={cn("rounded-md p-4 m-0 text-[0.75rem] leading-[1.6] bg-surface text-ink overflow-x-auto", plain && "whitespace-pre")}><code>{codeString}</code></pre>
      )}
      <button
        aria-label={plain ? "Copy diagram" : "Copy code"}
        onClick={copy}
        className="absolute top-2 right-2 rounded p-1 opacity-0 transition-opacity group-hover/code:opacity-100 hover:bg-surface-elevated/60 text-ink-subtle hover:text-ink"
      >
        {copied ? <Check className="h-3.5 w-3.5 text-ink" /> : <Copy className="h-3.5 w-3.5" />}
      </button>
    </div>
  )
}
