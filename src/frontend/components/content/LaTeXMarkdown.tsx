"use client"

import { MarkdownRenderer, type CodeRenderers, type MarkdownComponents } from "@gigachad/chat-ui"
import { memo, useMemo, type CSSProperties } from "react"
import { apiOrigin } from "@/lib/backend/urls"
import { CodeBlock } from "./CodeBlock"
import { CitationPill, type CitationSource } from "./diagrams/CitationPill"
import { MarkmapDiagram } from "./diagrams/MarkmapDiagram"
import { MermaidDiagram } from "./diagrams/MermaidDiagram"
import "katex/dist/katex.min.css"
import "@gigachad/chat-ui/markdown.css"

// Mermaid and Markmap are gigachad-only: the shared renderer routes fences to them, it does not know them.
const CODE_RENDERERS: CodeRenderers = {
  code: CodeBlock,
  mermaid: MermaidDiagram,
  markmap: MarkmapDiagram,
}

const SHELL_COMPONENTS: MarkdownComponents = {
  // Backend-emitted image paths (e.g. /mineru/images/...) are root-relative and
  // resolve against the backend, not whichever origin serves the frontend.
  img: ({ src, alt, ...props }) => (
    // eslint-disable-next-line @next/next/no-img-element -- backend/blob URLs resolved at runtime under Tauri static export, next/image can't optimize them
    <img alt={alt ?? ""} src={typeof src === "string" && src.startsWith("/") && !src.startsWith("//") ? `${apiOrigin()}${src}` : src} {...props} />
  ),
}

function LaTeXMarkdownInner({
  content,
  citationMap,
  streaming,
  compact,
  onContentChange,
  style,
}: {
  content: string
  citationMap?: Record<string, CitationSource>
  streaming?: boolean
  compact?: boolean
  onContentChange?: (newContent: string) => void
  style?: CSSProperties
}) {
  const components = useMemo<MarkdownComponents>(() => ({
    ...SHELL_COMPONENTS,
    a({ children, ...props }) {
      const info = citationMap?.[String(children)]
      return info ? <CitationPill source={info}>{children}</CitationPill> : <a {...props}>{children}</a>
    },
  }), [citationMap])

  return (
    <MarkdownRenderer
      content={content}
      streaming={streaming}
      compact={compact}
      style={style}
      codeRenderers={CODE_RENDERERS}
      components={components}
      onContentChange={onContentChange}
      checkboxClassName="mr-1.5 accent-ink cursor-pointer"
    />
  )
}

/** Render markdown with KaTeX math, citations, diagrams and optional checkbox toggling via `onContentChange`. */
export const LaTeXMarkdown = memo(LaTeXMarkdownInner)
