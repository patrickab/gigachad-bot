"use client"

import { Streamdown, defaultRemarkPlugins, type Components } from "streamdown"
import { createMathPlugin } from "@streamdown/math"
import remarkBreaks from "remark-breaks"
import { memo, useCallback, useContext, useEffect, useLayoutEffect, useMemo, useRef, useState, type ClipboardEvent as ReactClipboardEvent, type CSSProperties, type ReactNode } from "react"
import type { Markmap } from "markmap-view"
import { cn } from "@/lib/utils"
import { apiOrigin } from "@/lib/backend/urls"
import { fenceCode, selectionToMarkdown } from "@/components/content/selectionToMarkdown"
import { HostScaleContext } from "@/contexts/HostScaleContext"
import { offsetToCamera } from "@/lib/domain/viewport"
import { createPortal } from "react-dom"
import { Globe } from "lucide-react"
import { CodeBlock } from "./CodeBlock"
import { useThemeName } from "@/hooks/useThemeName"
import type { ThemeName } from "@/lib/palette"
import "katex/dist/katex.min.css"

const mathPlugin = createMathPlugin({ singleDollarTextMath: true })
const PLUGINS = { math: mathPlugin }

// Detect mermaid code in fences without a language tag by its diagram keyword.
function isMermaidCode(code: string): boolean {
  return /^\s*(graph|flowchart|sequenceDiagram|classDiagram|stateDiagram|erDiagram|gantt|pie|gitGraph|journey|requirementDiagram|mindmap|timeline|sankey|xychart|block-beta|packet-beta|architecture-beta)\b/i.test(code)
}

// ponytail: mutex serializes mermaid renders, it's a singleton that corrupts on concurrent calls
let mermaidQueue = Promise.resolve()

// Mirrors the globals.css role tokens (sRGB of the oklch palette). Hardcoded because mermaid
// can't parse oklch() and the browser preserves it through getComputedStyle/canvas. If the
// palette warmth is retuned in globals.css, regenerate these to match.
const MERMAID_DARK = {
  ink: "rgb(239, 234, 228)", inkMuted: "rgb(179, 173, 165)", inkSubtle: "rgb(130, 124, 118)",
  inkFaint: "rgb(98, 92, 88)", paper: "rgb(12, 8, 6)", surface: "rgb(21, 17, 14)",
  surfaceElevated: "rgb(32, 27, 23)",
}
const MERMAID_LIGHT = {
  ink: "rgb(80, 77, 70)", inkMuted: "rgb(108, 104, 97)", inkSubtle: "rgb(137, 134, 127)",
  inkFaint: "rgb(167, 164, 159)", paper: "rgb(245, 238, 224)", surface: "rgb(250, 246, 238)",
  surfaceElevated: "rgb(254, 252, 247)",
}

function enqueueMermaidRender(id: string, code: string): Promise<string> {
  const job = mermaidQueue.then(async () => {
    const mermaid = (await import("mermaid")).default
    const isLight =
      typeof document !== "undefined" && document.documentElement.classList.contains("light")
    const c = isLight ? MERMAID_LIGHT : MERMAID_DARK

    // theme: "base" is the only mermaid theme fully driven by themeVariables; named
    // themes recompute their own palette and ignore overrides. Re-initialize every
    // render (cheap, inside the mutex) so theme switches never inherit stale config.
    mermaid.initialize({
      startOnLoad: false,
      theme: "base",
      securityLevel: "strict",
      fontFamily: "var(--font-sans), system-ui, sans-serif",
      logLevel: "fatal",
      themeVariables: {
        primaryColor: c.surfaceElevated,
        primaryTextColor: c.ink,
        primaryBorderColor: c.inkFaint,
        secondaryColor: c.surface,
        secondaryTextColor: c.inkMuted,
        secondaryBorderColor: c.inkFaint,
        tertiaryColor: c.paper,
        tertiaryTextColor: c.inkSubtle,
        tertiaryBorderColor: c.inkFaint,
        lineColor: c.inkFaint,
        textColor: c.ink,
        mainBkg: c.surfaceElevated,
        nodeBorder: c.inkFaint,
        clusterBkg: c.surface,
        clusterBorder: c.inkFaint,
        edgeLabelBackground: c.paper,
        fontSize: "14px",
      },
    })
    await mermaid.parse(code)
    const { svg } = await mermaid.render(id, code)
    return svg
  })
  mermaidQueue = job.then(() => undefined, () => undefined)
  return job
}

function MermaidDiagram({ code }: { code: string }) {
  const [svg, setSvg] = useState("")
  const [failed, setFailed] = useState(false)
  const themeName = useThemeName()

  useEffect(() => {
    let cancelled = false
    setFailed(false)
    setSvg("")
    // Fresh id per render. Reusing one collides with the previous SVG still in the DOM
    // (re-render on theme switch), which makes mermaid.render fail.
    const id = `mmd-${Math.random().toString(36).slice(2)}`
    enqueueMermaidRender(id, code)
      .then((rendered) => { if (!cancelled) setSvg(rendered) })
      .catch(() => { if (!cancelled) setFailed(true) })
    return () => { cancelled = true }
  }, [code, themeName])

  if (failed) return <CodeBlock codeString={code} language="mermaid" />
  if (!svg) return <div className="my-3 text-xs text-ink-subtle">Rendering diagram…</div>

  return (
    <div
      data-markdown-source={fenceCode("mermaid", code)}
      className="my-3 flex justify-center overflow-x-auto [&_svg]:max-w-full [&_svg]:h-auto"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  )
}

const MIN_MARKMAP_SCALE = 0.5
const CANVAS_MARKMAP_PADDING = 24

function MarkmapDiagram({ code }: { code: string }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement | null>(null)
  const mmRef = useRef<Markmap | null>(null)
  const [failed, setFailed] = useState(false)
  const themeName = useThemeName()
  const [canvasViewportHeight, setCanvasViewportHeight] = useState<number | null>(null)
  const renderedRef = useRef<{ code: string; themeName: ThemeName } | null>(null)

  // A generated document sits under Streamdown's auto-sized <pre>/<code> wrappers.
  // Measure the canvas attachment itself so the map owns its complete viewport, not
  // the wrapper's 400px fallback.
  useLayoutEffect(() => {
    const attachment = containerRef.current?.closest<HTMLElement>("[data-canvas-attachment]")
    if (!attachment) return
    const measure = () => setCanvasViewportHeight(Math.max(0, attachment.clientHeight - CANVAS_MARKMAP_PADDING))
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(attachment)
    return () => observer.disconnect()
  }, [])

  // The map is a viewport pinned to its top-left corner, like React Flow. Keep
  // the world point at its center in the center when the frame resizes, and
  // when it sits in a canvas, magnify it with the canvas's zoom so the frame
  // shows the same view as the canvas zooms, just bigger or smaller.
  const hostScale = useContext(HostScaleContext)
  const hostScaleRef = useRef(hostScale)
  const viewportSizeRef = useRef({ width: 0, height: 0 })
  const reframe = useCallback((factor: number) => {
    const container = containerRef.current
    if (!container) return
    const next = { width: container.clientWidth, height: container.clientHeight }
    const previous = viewportSizeRef.current
    viewportSizeRef.current = next
    const mm = mmRef.current
    const transform = (svgRef.current as (SVGSVGElement & { __zoom?: { k: number, x: number, y: number } }) | null)?.__zoom
    if (!mm || !transform || previous.width === 0 || previous.height === 0) return
    const camera = offsetToCamera({ x: transform.x, y: transform.y }, transform.k, previous)
    mm.zoom.scaleTo(mm.svg, camera.scale * factor)
    mm.zoom.translateTo(mm.svg, camera.centerX, camera.centerY, [next.width / 2, next.height / 2])
  }, [])
  useEffect(() => {
    const container = containerRef.current
    if (!container || typeof ResizeObserver === "undefined") return
    const observer = new ResizeObserver(() => {
      const { width, height } = viewportSizeRef.current
      if (container.clientWidth !== width || container.clientHeight !== height) reframe(1)
    })
    observer.observe(container)
    return () => observer.disconnect()
  }, [reframe])
  useLayoutEffect(() => {
    const factor = hostScale / hostScaleRef.current
    hostScaleRef.current = hostScale
    if (factor !== 1) reframe(factor)
  }, [hostScale, reframe])

  useEffect(() => {
    let cancelled = false
    void (async () => {
      try {
        const { Transformer } = await import("markmap-lib")
        const { Markmap } = await import("markmap-view")
        if (cancelled || !containerRef.current) return
        if (containerRef.current.closest("[data-canvas-attachment]") && canvasViewportHeight === null) return
        if (renderedRef.current?.code === code && renderedRef.current.themeName === themeName) return

        const transformer = new Transformer()
        const { root } = transformer.transform(code)

        if (!svgRef.current) {
          const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg")
          svg.style.width = "100%"
          svg.style.height = "100%"
          containerRef.current.appendChild(svg)
          svgRef.current = svg
        }

        const isLight = themeName === "light"
        const ink = isLight ? "#1c1917" : MERMAID_DARK.ink
        const linkColor = isLight ? "#a8a29e" : MERMAID_DARK.inkFaint

        // Markmap's native drag and wheel controls make this a real viewport. Fit only
        // new content. A canvas resize must preserve the user's pan and zoom.
        if (mmRef.current) {
          await mmRef.current.setData(root)
        } else {
          mmRef.current = new Markmap(svgRef.current, {
            autoFit: false,
            duration: 300,
            paddingX: 16,
            initialExpandLevel: -1,
            zoom: true,
            pan: true,
            color: () => linkColor,
          })
          await mmRef.current.setData(root)
        }
        if (renderedRef.current?.code !== code) {
          try {
            await mmRef.current.fit()
            const scale = (svgRef.current as (SVGSVGElement & { __zoom?: { k?: number } })).__zoom?.k ?? 1
            if (scale < MIN_MARKMAP_SCALE) await mmRef.current.rescale(MIN_MARKMAP_SCALE / scale)
          } catch {}
        }
        renderedRef.current = { code, themeName }

        // text recolors on theme switch; link color is baked at create (minor)
        svgRef.current.style.setProperty("--markmap-text-color", ink)
      } catch {
        if (!cancelled) setFailed(true)
      }
    })()
    return () => { cancelled = true }
  }, [code, themeName, canvasViewportHeight])

  if (failed) return <CodeBlock codeString={code} language="markdown" />
  return <div ref={containerRef} data-markdown-source={fenceCode("markmap", code)} className="my-3 overflow-hidden rounded-md" style={{ height: canvasViewportHeight === null ? "var(--markmap-height, 400px)" : `${canvasViewportHeight}px` }} />
}

const REMARK_PLUGINS = [...Object.values(defaultRemarkPlugins), remarkBreaks]
const LINK_SAFETY_OFF = { enabled: false }

// Rewrite \( \) and \[ \] to $ and $$ outside code spans and fences.
function normalizeMathDelimiters(input: string): string {
  if (input.indexOf("\\(") === -1 && input.indexOf("\\[") === -1) return input
  const segments = input.split(/(```[\s\S]*?```|`[^`]*`)/g)
  return segments
    .map((seg, i) => {
      if (i % 2 === 1) return seg
      return seg
        .replace(/\\\[/g, () => "$$")
        .replace(/\\\]/g, () => "$$")
        .replace(/\\\(/g, () => "$")
        .replace(/\\\)/g, () => "$")
    })
    .join("")
}

// Box-drawing (U+2500-257F) and block element (U+2580-259F) chars: treat as a diagram when dense enough.
function containsAsciiTableArt(code: string): boolean {
  const boxChars = (code.match(/[─-╿▀-▟]/g) || []).length
  const lines = code.split("\n").length
  return boxChars >= 8 && lines >= 3
}


const PROSE_COMPONENTS: Components = {
  h1: "h1",
  h2: "h2",
  h3: "h3",
  h4: "h4",
  h5: "h5",
  h6: "h6",
  p: "p",
  ul: "ul",
  ol: "ol",
  li: "li",
  blockquote: "blockquote",
  hr: "hr",
  strong: "strong",
  em: "em",
  del: "del",
  // Backend-emitted image paths (e.g. /mineru/images/...) are root-relative and
  // resolve against the backend, not whichever origin serves the frontend.
  img: ({ src, ...props }) => (
    <img src={typeof src === "string" && src.startsWith("/") && !src.startsWith("//") ? `${apiOrigin()}${src}` : src} {...props} />
  ),
  table: ({ children, ...props }) => (
    <div className="overflow-x-auto my-3">
      <table {...props}>{children}</table>
    </div>
  ),
  thead: "thead",
  tbody: "tbody",
  tr: "tr",
  th: "th",
  td: "td",
  inlineCode: (props) => <code {...props} />,
  code: ({ className, children }) => {
    const match = /language-(\w+)/.exec(className || "")
    const codeString = String(children).replace(/\n$/, "")
    if (match?.[1] === "markmap" || className?.includes("markmap")) {
      return <MarkmapDiagram code={codeString} />
    }
    if (match?.[1] === "mermaid" || isMermaidCode(codeString)) {
      return <MermaidDiagram code={codeString} />
    }
    if (!match && containsAsciiTableArt(codeString)) {
      return <CodeBlock codeString={codeString} language="text" plain />
    }
    return <CodeBlock codeString={codeString} language={match?.[1] || "text"} />
  },
}

// Hover pill for numeric citation links. The popup is portaled and hides after a short delay so the pointer can reach it.
function CitationPill({ source, children }: { source: { title: string; url: string; content: string }; children: ReactNode }) {
  const pillRef = useRef<HTMLAnchorElement>(null)
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null)
  const hideTimer = useRef<ReturnType<typeof setTimeout>>(null)

  const show = useCallback(() => {
    if (hideTimer.current) { clearTimeout(hideTimer.current); hideTimer.current = null }
    const el = pillRef.current
    if (!el) return
    const r = el.getBoundingClientRect()
    setPos({ top: r.top - 8, left: r.left + r.width / 2 })
  }, [])

  const hide = useCallback(() => {
    hideTimer.current = setTimeout(() => setPos(null), 150)
  }, [])

  return (
    <>
      <a
        ref={pillRef}
        href={source.url}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-flex items-center align-middle rounded-full bg-surface-elevated/40 border border-divider-strong/40 px-1.5 py-0 text-[10px] text-ink-muted hover:bg-hover hover:border-divider-strong transition-colors no-underline"
        onMouseEnter={show}
        onMouseLeave={hide}
      >
        {children}
      </a>
      {pos && createPortal(
        <div
          style={{ position: "fixed", top: pos.top, left: pos.left, transform: "translate(-50%, -100%)", zIndex: 9999 }}
          className="w-64 rounded-md border border-divider bg-surface p-2.5 shadow-[var(--shadow-xl)]"
          onMouseEnter={show}
          onMouseLeave={hide}
        >
          <span className="flex items-center gap-1.5 mb-0.5">
            <Globe className="h-2.5 w-2.5 text-ink shrink-0" />
            <span className="text-[10px] font-medium text-ink truncate">{children}</span>
          </span>
          {source.title && (
            <span className="text-[10px] font-medium text-ink mb-0.5 line-clamp-2 block">{source.title}</span>
          )}
          {source.content && (
            <span className="text-[9px] text-ink-subtle line-clamp-3 block">{source.content}</span>
          )}
        </div>,
        document.body
      )}
    </>
  )
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
  citationMap?: Record<string, { title: string; url: string; content: string }>
  streaming?: boolean
  compact?: boolean
  onContentChange?: (newContent: string) => void
  style?: CSSProperties
}) {
  const processed = useMemo(() => normalizeMathDelimiters(content), [content])

  // Source line index of each task item, only tracked when toggling is enabled.
  const checkboxLines = useMemo(() => {
    if (!onContentChange) return []
    const lines = content.split("\n")
    const indices: number[] = []
    for (let i = 0; i < lines.length; i++) {
      if (/^- \[[ x]\]/.test(lines[i].trimStart())) indices.push(i)
    }
    return indices
  }, [content, !!onContentChange])

  const handleToggle = useCallback((idx: number, wasChecked: boolean) => {
    if (!onContentChange) return
    const lineIdx = checkboxLines[idx]
    if (lineIdx === undefined) return
    const lines = content.split("\n")
    lines[lineIdx] = wasChecked
      ? lines[lineIdx].replace("- [x]", "- [ ]")
      : lines[lineIdx].replace("- [ ]", "- [x]")
    onContentChange(lines.join("\n"))
  }, [content, checkboxLines, onContentChange])

  // Render-order counter mapping each rendered checkbox to its source line in `checkboxLines`.
  const checkboxIdx = useRef(0)
  checkboxIdx.current = 0

  const components = useMemo<Components>(() => {
    const base: Components = {
      ...PROSE_COMPONENTS,
      a({ children, href, title, ...props }) {
        const num = String(children)
        const info = citationMap?.[num]
        if (info) {
          return (
            <CitationPill source={info}>
              {children}
            </CitationPill>
          )
        }
        return (
          <a
            href={href}
            title={title}
            className="text-ink-muted hover:text-ink underline underline-offset-2 decoration-divider-strong hover:decoration-ink-muted transition-colors"
            {...props}
          >
            {children}
          </a>
        )
      },
    }
    if (onContentChange) {
      base.input = ({ checked, type, ...rest }) => {
        if (type !== "checkbox") return <input type={type} checked={checked} {...rest} />
        const idx = checkboxIdx.current++
        return (
          <input
            type="checkbox"
            checked={checked}
            className="mr-1.5 accent-ink cursor-pointer"
            onChange={() => handleToggle(idx, !!checked)}
          />
        )
      }
    }
    return base
  }, [citationMap, onContentChange, handleToggle])

  // Copy the underlying markdown (headings, math, code fences) instead of the rendered text.
  const handleCopy = useCallback((event: ReactClipboardEvent<HTMLDivElement>) => {
    const selection = window.getSelection()
    const markdown = selection && selectionToMarkdown(selection, event.currentTarget)
    if (!markdown) return
    event.preventDefault()
    event.clipboardData.setData("text/plain", markdown)
  }, [])

  return (
    <div
      className={cn("text-[13px] leading-relaxed", compact ? "markdown-body-compact" : "markdown-body")}
      style={style}
      onCopy={handleCopy}
    >
      <Streamdown
        mode={streaming ? "streaming" : "static"}
        isAnimating={false}
        animated={false}
        caret={undefined}
        controls={{ code: false, table: { copy: true, download: true }, mermaid: false }}
        plugins={PLUGINS}
        remarkPlugins={REMARK_PLUGINS}
        linkSafety={LINK_SAFETY_OFF}
        components={components}
      >
        {processed}
      </Streamdown>
    </div>
  )
}

/** Render markdown with KaTeX math, citations, diagrams and optional checkbox toggling via `onContentChange`. */
export const LaTeXMarkdown = memo(LaTeXMarkdownInner)
