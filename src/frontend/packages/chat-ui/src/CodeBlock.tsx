"use client"

import { Fragment, type CSSProperties } from "react"
import { Check, Copy } from "lucide-react"
import type { CodeBlockProps } from "./components.tsx"
import { useCopyFeedback } from "./useCopyFeedback.ts"
import { useHighlightedHtml } from "./useHighlightedHtml.ts"

/** Color scheme of the code and diagram renderers. The host passes its own, the package reads no theme state. */
export type ColorScheme = "light" | "dark"

/** Props of the package's fenced-code renderers: the fence plus the host's color scheme. */
export interface ThemedCodeBlockProps extends CodeBlockProps {
  theme: ColorScheme
}

/** Syntax-highlighted code under a header with its language and a copy button, lines numbered. Shared by
 *  markdown-rendered fenced code and any other surface that needs to show a script (e.g. a tool call's source).
 *  `plain` skips highlighting and line numbers, for diagrams drawn with text characters. Styled by `code.css`. */
export function CodeBlock({ codeString, language, plain = false, theme }: ThemedCodeBlockProps) {
  const { copied, copy } = useCopyFeedback(codeString)
  // Highlighting is async, so the plain `<pre>` shows until it resolves.
  const html = useHighlightedHtml(codeString, language, !plain)
  const lines = codeString.split("\n")
  const gutter = { "--gc-code-gutter": `${String(lines.length).length}ch` } as CSSProperties

  return (
    <div className={plain ? "gc-code gc-code-plain" : "gc-code"} data-theme={theme} data-code-language={language === "text" ? "" : language} style={gutter}>
      <div className="gc-code-header">
        <span className="gc-code-language">{plain ? "" : language}</span>
        <button type="button" aria-label={plain ? "Copy diagram" : "Copy code"} onClick={copy} className="gc-code-copy">
          {copied ? <Check size={14} aria-hidden="true" className="gc-code-copied" /> : <Copy size={14} aria-hidden="true" />}
        </button>
      </div>
      {html ? (
        <div className="gc-code-highlighted" dangerouslySetInnerHTML={{ __html: html }} />
      ) : (
        // Same `.line` spans as Shiki's output, so line numbers show before highlighting and nothing shifts.
        <pre className="gc-code-fallback"><code>{lines.map((line, index) => (
          <Fragment key={index}>{index > 0 && "\n"}<span className="line">{line}</span></Fragment>
        ))}</code></pre>
      )}
    </div>
  )
}
