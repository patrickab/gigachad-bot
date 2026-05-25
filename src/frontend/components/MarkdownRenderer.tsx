"use client"

import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import remarkMath from "remark-math"
import rehypeKatex from "rehype-katex"
import rehypeRaw from "rehype-raw"
import type { Components } from "react-markdown"
import { memo } from "react"
import { Light as SyntaxHighlighter } from "react-syntax-highlighter"
import latex from "react-syntax-highlighter/dist/cjs/languages/hljs/latex"
import { atomOneDark } from "react-syntax-highlighter/dist/cjs/styles/hljs"

SyntaxHighlighter.registerLanguage("latex", latex)

const components: Components = {
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

function MarkdownRendererInner({ content }: { content: string }) {
  return (
    <div className="markdown-body text-sm leading-relaxed">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeRaw]}
        components={components}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}

export const MarkdownRenderer = memo(MarkdownRendererInner)
