"use client"

import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import remarkMath from "remark-math"
import remarkBreaks from "remark-breaks"
import rehypeKatex from "rehype-katex"
import rehypeRaw from "rehype-raw"
import type { Components } from "react-markdown"
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Light as SyntaxHighlighter } from "react-syntax-highlighter"
import latex from "react-syntax-highlighter/dist/cjs/languages/hljs/latex"
import { atomOneDark } from "react-syntax-highlighter/dist/cjs/styles/hljs"
import { motion, AnimatePresence } from "framer-motion"
import { Globe } from "lucide-react"
import "katex/dist/katex.min.css"

SyntaxHighlighter.registerLanguage("latex", latex)

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
        className="inline-flex items-center rounded-full bg-zinc-800/40 border border-zinc-700/40 px-1.5 py-0 text-[10px] text-zinc-400 hover:bg-zinc-700/50 hover:border-zinc-600/50 transition-colors no-underline"
      >
        {children}
      </a>
      <AnimatePresence>
        {show && (
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.9, opacity: 0 }}
            transition={{ duration: 0.15, ease: "easeOut" }}
            onMouseEnter={handleEnter}
            onMouseLeave={handleLeave}
            className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 w-64 rounded-md border border-zinc-800 bg-zinc-900 p-2.5 shadow-xl z-50"
          >
            <div className="flex items-center gap-1.5 mb-0.5">
              <Globe className="h-2.5 w-2.5 text-sky-400 shrink-0" />
              <span className="text-[10px] font-medium text-sky-400 truncate">{children}</span>
            </div>
            {popupTitle && (
              <div className="text-[10px] font-medium text-zinc-300 mb-0.5 line-clamp-2">{popupTitle}</div>
            )}
            {popupContent && (
              <div className="text-[9px] text-zinc-500 line-clamp-3">{popupContent}</div>
            )}
            <div className="absolute left-1/2 -translate-x-1/2 -bottom-1 w-2 h-2 rotate-45 bg-zinc-900 border-r border-b border-zinc-800" />
          </motion.div>
        )}
      </AnimatePresence>
    </span>
  )
}

const sharedComponents: Components = {
  pre({ children }) {
    return <pre>{children}</pre>
  },
  code({ className, children, ...props }) {
    const match = /language-(\w+)/.exec(className || "")
    const isInline = !match && !className
    if (isInline) {
      return (
        <code className={className} {...props}>
          {children}
        </code>
      )
    }
    const codeString = String(children).replace(/\n$/, "")
    return (
      <SyntaxHighlighter
        language={match ? match[1] : "text"}
        style={atomOneDark}
        customStyle={{
          margin: 0,
          borderRadius: "0.5rem",
          fontSize: "0.75rem",
          lineHeight: "1.6",
        }}
        PreTag="div"
      >
        {codeString}
      </SyntaxHighlighter>
    )
  },
  table({ children }) {
    return (
      <div className="overflow-x-auto">
        <table>{children}</table>
      </div>
    )
  },
}

function LaTeXMarkdownInner({ content, citationMap }: { content: string; citationMap?: Record<string, { title: string; url: string; content: string }> }) {
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
        <a href={href} title={title} className="text-sky-400 hover:text-sky-300 underline underline-offset-2" {...props}>
          {children}
        </a>
      )
    },
  }), [citationMap])

  return (
    <div className="markdown-body text-sm leading-relaxed">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath, remarkBreaks]}
        rehypePlugins={[rehypeKatex, rehypeRaw]}
        components={allComponents}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}

export const LaTeXMarkdown = memo(LaTeXMarkdownInner)