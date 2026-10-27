"use client"

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

/** Syntax-highlighted code with a hover-revealed copy button. Shared by markdown-rendered
 *  fenced code and any other surface that needs to show a script (e.g. a tool call's source).
 *  `plain` skips highlighting, for diagrams drawn with text characters. Styled by `code.css`. */
export function CodeBlock({ codeString, language, plain = false, theme }: ThemedCodeBlockProps) {
  const { copied, copy } = useCopyFeedback(codeString)
  // Highlighting is async, so the plain `<pre>` shows until it resolves.
  const html = useHighlightedHtml(codeString, language, !plain)

  return (
    <div className={plain ? "gc-code gc-code-plain" : "gc-code"} data-theme={theme} data-code-language={language === "text" ? "" : language}>
      {html ? (
        <div className="gc-code-highlighted" dangerouslySetInnerHTML={{ __html: html }} />
      ) : (
        <pre className="gc-code-fallback"><code>{codeString}</code></pre>
      )}
      <button aria-label={plain ? "Copy diagram" : "Copy code"} onClick={copy} className="gc-code-copy">
        {copied ? <Check className="gc-code-copied" /> : <Copy />}
      </button>
    </div>
  )
}
