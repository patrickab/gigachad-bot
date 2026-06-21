"use client"

import { Streamdown, defaultRemarkPlugins, type Components } from "streamdown"
import { createMathPlugin } from "@streamdown/math"
import remarkBreaks from "remark-breaks"
import { cloneElement, isValidElement, memo, useCallback, useEffect, useMemo, useRef, useState } from "react"
import { cn } from "@/lib/utils"
import { highlightCode } from "@/lib/markdown-syntax-highlighting"
import { createPortal } from "react-dom"
import { Check, Copy, Globe } from "lucide-react"
import "katex/dist/katex.min.css"

const mathPlugin = createMathPlugin({ singleDollarTextMath: true })
const PLUGINS = { math: mathPlugin }

function isMermaidCode(code: string): boolean {
  return /^\s*(graph|flowchart|sequenceDiagram|classDiagram|stateDiagram|erDiagram|gantt|pie|gitGraph|journey|requirementDiagram|mindmap|timeline|sankey|xychart|block-beta|packet-beta|architecture-beta)\b/i.test(code)
}

function MermaidDiagram({ code }: { code: string }) {
  const [svg, setSvg] = useState("")
  const [failed, setFailed] = useState(false)
  const idRef = useRef(`mmd-${Math.random().toString(36).slice(2)}`)

  useEffect(() => {
    let cancelled = false
    setFailed(false)
    setSvg("")
    void (async () => {
      try {
        const mermaid = (await import("mermaid")).default
        const isLight =
          typeof document !== "undefined" && document.documentElement.classList.contains("light")
        mermaid.initialize({
          startOnLoad: false,
          theme: isLight ? "neutral" : "dark",
          securityLevel: "strict",
          fontFamily: "var(--font-sans), system-ui, sans-serif",
          logLevel: "fatal",
        })
        await mermaid.parse(code)
        const { svg: rendered } = await mermaid.render(idRef.current, code)
        if (!cancelled) setSvg(rendered)
      } catch {
        if (!cancelled) setFailed(true)
      }
    })()
    return () => { cancelled = true }
  }, [code])

  if (failed) return <CodeBlock codeString={code} language="mermaid" />
  if (!svg) return <div className="my-3 text-xs text-ink-subtle">Rendering diagram…</div>

  return (
    <div
      className="my-3 flex justify-center overflow-x-auto [&_svg]:max-w-full [&_svg]:h-auto"
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  )
}

const REMARK_PLUGINS = [...Object.values(defaultRemarkPlugins), remarkBreaks]
const LINK_SAFETY_OFF = { enabled: false }

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

function AsciiDiagram({ codeString }: { codeString: string }) {
  const [copied, setCopied] = useState(false)
  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(codeString).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    })
  }, [codeString])

  return (
    <div className="group/code relative my-3">
      <pre className="rounded-md p-4 m-0 text-[0.75rem] leading-[1.6] bg-surface text-ink overflow-x-auto whitespace-pre">
        <code>{codeString}</code>
      </pre>
      <button
        onClick={handleCopy}
        className="absolute top-2 right-2 rounded p-1 opacity-0 transition-opacity group-hover/code:opacity-100 hover:bg-surface-elevated/60 text-ink-subtle hover:text-ink"
        title="Copy code"
      >
        {copied ? <Check className="h-3.5 w-3.5 text-ink" /> : <Copy className="h-3.5 w-3.5" />}
      </button>
    </div>
  )
}

function containsAsciiTableArt(code: string): boolean {
  const boxChars = (code.match(/[─-╿▀-▟]/g) || []).length
  const lines = code.split("\n").length
  return boxChars >= 8 && lines >= 3
}

function CodeBlock({ codeString, language }: { codeString: string; language: string }) {
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
        title="Copy code"
      >
        {copied ? <Check className="h-3.5 w-3.5 text-ink" /> : <Copy className="h-3.5 w-3.5" />}
      </button>
    </div>
  )
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
  img: "img",
  table: "table",
  thead: "thead",
  tbody: "tbody",
  tr: "tr",
  th: "th",
  td: "td",
  pre: ({ children }: any) =>
    isValidElement(children)
      ? cloneElement(children, { "data-block": "true" } as Record<string, string>)
      : <>{children}</>,
  code: ({ className, children, ...props }: any) => {
    if (!("data-block" in props)) {
      return (
        <code className={className} {...props}>
          {children}
        </code>
      )
    }
    const match = /language-(\w+)/.exec(className || "")
    const codeString = String(children).replace(/\n$/, "")
    if (match?.[1] === "mermaid" || isMermaidCode(codeString)) {
      return <MermaidDiagram code={codeString} />
    }
    if (!match && containsAsciiTableArt(codeString)) {
      return <AsciiDiagram codeString={codeString} />
    }
    return <CodeBlock codeString={codeString} language={match?.[1] || "text"} />
  },
}

function CitationPill({ children, href, title, ...props }: any) {
  const pillRef = useRef<HTMLAnchorElement>(null)
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null)
  const hideTimer = useRef<ReturnType<typeof setTimeout>>(null)

  const popupTitle = (props["data-title"] as string) || title || ""
  const popupContent = (props["data-content"] as string) || ""

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
        href={href}
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
          {popupTitle && (
            <span className="text-[10px] font-medium text-ink mb-0.5 line-clamp-2 block">{popupTitle}</span>
          )}
          {popupContent && (
            <span className="text-[9px] text-ink-subtle line-clamp-3 block">{popupContent}</span>
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
}: {
  content: string
  citationMap?: Record<string, { title: string; url: string; content: string }>
  streaming?: boolean
  compact?: boolean
  onContentChange?: (newContent: string) => void
}) {
  const processed = useMemo(() => normalizeMathDelimiters(content), [content])

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

  const checkboxIdx = useRef(0)
  checkboxIdx.current = 0

  const components = useMemo<Components>(() => {
    const base: Components = {
      ...PROSE_COMPONENTS,
      a({ children, href, title, ...props }: any) {
        const num = String(children)
        const info = citationMap?.[num]
        if (info && /^\d+$/.test(num)) {
          return (
            <CitationPill
              href={info.url}
              title={info.title}
              data-title={info.title}
              data-content={info.content}
            >
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
      base.input = ({ checked, type, ...rest }: any) => {
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

  return (
    <div className={cn("text-[13px] leading-relaxed", compact ? "markdown-body-compact" : "markdown-body")}>
      <Streamdown
        mode={streaming ? "streaming" : "static"}
        isAnimating={!!streaming}
        animated={!!streaming}
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

export const LaTeXMarkdown = memo(LaTeXMarkdownInner)
