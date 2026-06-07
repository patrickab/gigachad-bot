"use client"

import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import remarkMath from "remark-math"
import remarkBreaks from "remark-breaks"
import rehypeKatex from "rehype-katex"
import rehypeRaw from "rehype-raw"
import type { Components } from "react-markdown"
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react"
import { cn } from "@/lib/utils"
import { highlightCode } from "@/lib/markdown-syntax-highlighting"
import { motion, AnimatePresence } from "framer-motion"
import { Check, Copy, Globe } from "lucide-react"
import "katex/dist/katex.min.css"

const FULL_REMARK_PLUGINS = [remarkGfm, remarkMath, remarkBreaks]
const FULL_REHYPE_PLUGINS = [rehypeKatex, rehypeRaw]
const STREAMING_REMARK_PLUGINS = [remarkGfm, remarkBreaks]
const STREAMING_REHYPE_PLUGINS = [rehypeRaw]

const streamingComponents: Components & { thought?: React.ComponentType<any> } = {
  pre({ children }) {
    return <>{children}</>
  },
  code({ className, children, ...props }) {
    return (
      <code className={className} {...props}>
        {children}
      </code>
    )
  },
  table({ children }) {
    return (
      <div className="overflow-x-auto">
        <table>{children}</table>
      </div>
    )
  },
  thought() {
    return null
  },
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

function CitationPill({ node, children, href, title, ...props }: any) {
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
            className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 rounded-md border border-divider bg-surface p-2.5 shadow-xl z-50 block"
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

const sharedComponents: Components & { thought?: React.ComponentType<any> } = {
  pre({ children }) {
    return <>{children}</>
  },
  code({ className, children, node, ...props }) {
    const match = /language-(\w+)/.exec(className || "")
    if (!match && !className) {
      return (
        <code className={className} {...props}>
          {children}
        </code>
      )
    }
    const language = match ? match[1] : "text"
    const codeString = String(children).replace(/\n$/, "")
    return <CodeBlock codeString={codeString} language={language} />
  },
  table({ children }) {
    return (
      <div className="overflow-x-auto">
        <table>{children}</table>
      </div>
    )
  },
  thought() {
    return null
  },
}

function LaTeXMarkdownInner({ content, citationMap, streaming, compact, onContentChange }: { content: string; citationMap?: Record<string, { title: string; url: string; content: string }>; streaming?: boolean; compact?: boolean; onContentChange?: (newContent: string) => void }) {
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
  if (onContentChange) checkboxIdx.current = 0

  const allComponents = useMemo<Components>(() => ({
    ...sharedComponents,
    a({ node, children, href, title, ...props }: any) {
      const num = String(children)
      const info = citationMap?.[num]
      if (info && /^\d+$/.test(num)) {
        return (
          <CitationPill
            node={node}
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
        <a href={href} title={title} className="text-ink hover:text-ink underline underline-offset-2" {...props}>
          {children}
        </a>
      )
    },
  }), [citationMap])

  const interactiveInput = onContentChange ? {
    input({ checked, type, ...rest }: any) {
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
    },
  } : null

  const remarkPlugins = streaming
    ? STREAMING_REMARK_PLUGINS
    : FULL_REMARK_PLUGINS

  const rehypePlugins = streaming
    ? STREAMING_REHYPE_PLUGINS
    : FULL_REHYPE_PLUGINS

  const baseComponents = streaming ? streamingComponents : allComponents
  const components = interactiveInput ? { ...baseComponents, ...interactiveInput } : baseComponents

  return (
    <div className={cn("text-sm leading-relaxed", compact ? "markdown-body-compact" : "markdown-body")} style={{ color: "var(--ink)" }}>
      <ReactMarkdown
        remarkPlugins={remarkPlugins}
        rehypePlugins={rehypePlugins}
        components={components}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}

export const LaTeXMarkdown = memo(LaTeXMarkdownInner)
