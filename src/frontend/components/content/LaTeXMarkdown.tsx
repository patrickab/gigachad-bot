"use client"

import { CodeBlock, MarkdownRenderer, MarkmapDiagram, MermaidDiagram, type CodeBlockProps, type CodeRenderers, type MarkdownComponents, type MarkmapViewport } from "@gigachad/chat-ui"
import { memo, useContext, useMemo, useRef, type CSSProperties, type RefObject } from "react"
import { HostScaleContext } from "@/contexts/HostScaleContext"
import { useThemeName } from "@/hooks/useThemeName"
import { useViewportReframe } from "@/hooks/useViewportReframe"
import { apiOrigin } from "@/lib/backend/urls"
import { offsetToCamera } from "@/lib/domain/viewport"
import { CitationPill, type CitationSource } from "./diagrams/CitationPill"
import "katex/dist/katex.min.css"
import "@gigachad/chat-ui/markdown.css"
import "@gigachad/chat-ui/code.css"

function ShellCodeBlock(props: CodeBlockProps) {
  return <CodeBlock {...props} theme={useThemeName()} />
}

function ShellMermaid({ code }: { code: string }) {
  return <MermaidDiagram code={code} theme={useThemeName()} />
}

/** A markmap in the shell's theme. Inside a canvas it magnifies with the canvas's zoom, and it keeps the world point at its center centered when its frame resizes. */
function ShellMarkmap({ code }: { code: string }) {
  const viewport = useRef<MarkmapViewport>(null)
  const frame = useMemo<RefObject<HTMLElement | null>>(() => ({ get current() { return viewport.current?.element ?? null } }), [])
  useViewportReframe(frame, {
    hostScale: useContext(HostScaleContext),
    ready: true,
    skipFromEmpty: true,
    read: () => viewport.current?.camera() ?? null,
    write: (offset, scale, size) => viewport.current?.setCamera(offsetToCamera(offset, scale, size), size),
  })
  return <MarkmapDiagram code={code} theme={useThemeName()} viewportRef={viewport} />
}

// The shared renderer routes fences to the package's renderers, these supply the shell's theme.
const CODE_RENDERERS: CodeRenderers = {
  code: ShellCodeBlock,
  mermaid: ShellMermaid,
  markmap: ShellMarkmap,
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
