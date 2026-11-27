"use client"

import { CodeBlock, MarkdownRenderer, MarkmapDiagram, MermaidDiagram, type CodeBlockProps, type CodeRenderers, type MarkdownComponents, type MarkmapViewport } from "@gigachad/chat-ui"
import { memo, useContext, useMemo, useRef, type CSSProperties, type RefObject } from "react"
import { HostScaleContext } from "@/contexts/HostScaleContext"
import { useThemeName } from "@/hooks/useThemeName"
import { useViewportReframe } from "@/hooks/useViewportReframe"
import { apiRoot, fileViewerRawUrl } from "@/lib/backend/urls"
import { offsetToCamera } from "@/lib/domain/viewport"
import { remarkInlineCodeFileLinks, resolveLocalFileHref } from "@/lib/fileLinks"
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

/** A path relative to the document (`images/a.png`, `./a.png`, `../a.png`): no scheme, not root-relative, no fragment. */
function isRelativePath(src: string): boolean {
  return src !== "" && !/^[a-z][a-z\d+.-]*:/i.test(src) && !src.startsWith("/") && !src.startsWith("#")
}

/** `src` as the browser loads it: backend-emitted root-relative paths (e.g. /mineru/images/...) against the backend, not whichever origin serves the frontend, relative ones against `imageDirectory` through the file API. */
function imageUrl(src: unknown, imageDirectory: string | undefined): unknown {
  if (typeof src !== "string") return src
  if (src.startsWith("/") && !src.startsWith("//")) return `${apiRoot()}${src}`
  if (imageDirectory !== undefined && isRelativePath(src)) return fileViewerRawUrl(`${imageDirectory}/${src}`)
  return src
}

function imageComponent(imageDirectory: string | undefined): MarkdownComponents["img"] {
  return function Image({ src, alt, ...props }) {
    // eslint-disable-next-line @next/next/no-img-element -- backend/blob URLs resolved at runtime, next/image can't optimize them
    return <img alt={alt ?? ""} src={imageUrl(src, imageDirectory) as typeof src} {...props} />
  }
}

const FILE_LINK_PLUGINS = [remarkInlineCodeFileLinks]

/** Local file links of an agent answer: relative ones resolve against `base`, `open` shows the absolute path. */
export interface FileLinks {
  base: string
  /** Opens `path`, at 1-based `line` where the opener can (a finding's `file:line`). */
  open: (path: string, line?: number) => void
}

function LaTeXMarkdownInner({
  content,
  streaming,
  compact,
  onContentChange,
  style,
  imageDirectory,
  fileLinks,
}: {
  content: string
  streaming?: boolean
  compact?: boolean
  onContentChange?: (newContent: string) => void
  style?: CSSProperties
  /** Host folder relative image paths resolve against, the document's own. Without it they stay blocked. */
  imageDirectory?: string
  /** Opens local file links and inline-code paths in the shell instead of navigating. */
  fileLinks?: FileLinks
}) {
  const components = useMemo<MarkdownComponents>(() => ({
    img: imageComponent(imageDirectory),
    a({ children, ...props }) {
      const path = fileLinks && props.href ? resolveLocalFileHref(props.href, fileLinks.base) : null
      if (path && fileLinks) {
        return <a {...props} title={path} onClick={(event) => { event.preventDefault(); fileLinks.open(path) }}>{children}</a>
      }
      return <a {...props}>{children}</a>
    },
  }), [imageDirectory, fileLinks])

  return (
    <MarkdownRenderer
      content={content}
      streaming={streaming}
      compact={compact}
      style={style}
      codeRenderers={CODE_RENDERERS}
      components={components}
      remarkPlugins={fileLinks ? FILE_LINK_PLUGINS : undefined}
      hardenLinks={imageDirectory === undefined && !fileLinks}
      onContentChange={onContentChange}
      checkboxClassName="mr-1.5 accent-ink cursor-pointer"
    />
  )
}

/** Render markdown with KaTeX math, citations, diagrams and optional checkbox toggling via `onContentChange`. */
export const LaTeXMarkdown = memo(LaTeXMarkdownInner)
