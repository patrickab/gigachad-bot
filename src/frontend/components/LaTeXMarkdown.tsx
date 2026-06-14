"use client"

import { Streamdown, type Components } from "streamdown"
import { createMathPlugin } from "@streamdown/math"
import remarkBreaks from "remark-breaks"
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react"
import { cn } from "@/lib/utils"
import { highlightCode } from "@/lib/markdown-syntax-highlighting"
import { motion, AnimatePresence } from "framer-motion"
import { Check, Copy, Globe } from "lucide-react"
import "katex/dist/katex.min.css"

// Streamdown plugins are stateful singletons (they lazily wire up KaTeX), so
// create once at module scope. Code blocks are rendered by our own minimal
// CodeBlock (below) instead of @streamdown/code: it keeps the bare,
// chrome-free look of the rest of the app and, unlike the tokenized streamdown
// renderer, preserves newlines so multi-line ASCII diagrams render correctly.
const mathPlugin = createMathPlugin({ singleDollarTextMath: false })
const PLUGINS = { math: mathPlugin }

const REMARK_PLUGINS = [remarkBreaks]
const LINK_SAFETY_OFF = { enabled: false }

// remark-math (used by @streamdown/math) only understands $...$ / $$...$$.
// LLMs frequently emit \(...\) and \[...\], so normalize those to dollars.
// Fenced and inline code segments are skipped so code samples aren't mangled.
function normalizeMathDelimiters(input: string): string {
  if (input.indexOf("\\(") === -1 && input.indexOf("\\[") === -1) return input
  const segments = input.split(/(```[\s\S]*?```|`[^`]*`)/g)
  return segments
    .map((seg, i) => {
      if (i % 2 === 1) return seg // code segment, leave untouched
      return seg
        .replace(/\\\[/g, () => "$$")
        .replace(/\\\]/g, () => "$$")
        .replace(/\\\(/g, () => "$")
        .replace(/\\\)/g, () => "$")
    })
    .join("")
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

// Prose elements are rendered as bare intrinsic tags so they shed Streamdown's
// built-in Tailwind classes and inherit the app's .markdown-body token theme.
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
  thead: "thead",
  tbody: "tbody",
  tr: "tr",
  th: "th",
  td: "td",
  img: "img",
  table: ({ children }: any) => (
    <div className="overflow-x-auto">
      <table>{children}</table>
    </div>
  ),
  pre: ({ children }: any) => <>{children}</>,
  code: ({ className, children, ...props }: any) => {
    const match = /language-(\w+)/.exec(className || "")
    if (!match) {
      return (
        <code className={className} {...props}>
          {children}
        </code>
      )
    }
    const codeString = String(children).replace(/\n$/, "")
    return <CodeBlock codeString={codeString} language={match[1]} />
  },
}

function CitationPill({ children, href, title, ...props }: any) {
  const [show, setShow] = useState(false)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const handleEnter = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => setShow(true), 80)
  }, [])

  const handleLeave = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => setShow(false), 150)
  }, [])

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current)
    }
  }, [])

  const popupTitle = (props["data-title"] as string) || title || ""
  const popupContent = (props["data-content"] as string) || ""

  return (
    <span className="relative inline-flex items-center align-middle">
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        onMouseEnter={handleEnter}
        onMouseLeave={handleLeave}
        className="inline-flex items-center rounded-full bg-surface-elevated/40 border border-divider-strong/40 px-1.5 py-0 text-[10px] text-ink-muted hover:bg-hover hover:border-divider-strong transition-colors no-underline"
      >
        {children}
      </a>
      <AnimatePresence>
        {show && (
          <motion.span
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            transition={{ duration: 0.15, ease: "easeOut" }}
            onMouseEnter={handleEnter}
            onMouseLeave={handleLeave}
            className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 rounded-md border border-divider bg-surface p-2.5 shadow-[var(--shadow-xl)] z-50 block"
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
            <span className="absolute left-1/2 -translate-x-1/2 -bottom-1 w-2 h-2 rotate-45 bg-surface border-r border-b border-divider block" />
          </motion.span>
        )}
      </AnimatePresence>
    </span>
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
    <div
      className={cn("text-sm leading-relaxed", compact ? "markdown-body-compact" : "markdown-body")}
      style={{ color: "var(--ink)" }}
    >
      <Streamdown
        mode={streaming ? "streaming" : "static"}
        isAnimating={!!streaming}
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
