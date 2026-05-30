"use client"

import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import remarkMath from "remark-math"
import remarkBreaks from "remark-breaks"
import rehypeKatex from "rehype-katex"
import rehypeRaw from "rehype-raw"
import type { Components } from "react-markdown"
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react"
import { highlightCode } from "@/lib/shiki-highlighter"
import { motion, AnimatePresence } from "framer-motion"
import { Check, Copy, Globe } from "lucide-react"
import "katex/dist/katex.min.css"

function CodeBlock({ codeString, language }: { codeString: string; language: string }) {
  const [copied, setCopied] = useState(false)
  const html = useMemo(() => highlightCode(codeString, language), [codeString, language])

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
        <pre className="rounded-md p-4 m-0 text-[0.75rem] leading-[1.6] bg-[#1e1e1e] text-[#d4d4d4] overflow-x-auto"><code>{codeString}</code></pre>
      )}
      <button
        onClick={handleCopy}
        className="absolute top-2 right-2 rounded p-1 opacity-0 transition-opacity group-hover/code:opacity-100 hover:bg-zinc-700/60 text-zinc-500 hover:text-zinc-300"
        title="Copy code"
      >
        {copied ? <Check className="h-3.5 w-3.5 text-cyan-400" /> : <Copy className="h-3.5 w-3.5" />}
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