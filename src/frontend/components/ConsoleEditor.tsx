"use client"

import { useCallback, useRef } from "react"
import { Light as SyntaxHighlighter } from "react-syntax-highlighter"
import latex from "react-syntax-highlighter/dist/cjs/languages/hljs/latex"
import { atomOneDark } from "react-syntax-highlighter/dist/cjs/styles/hljs"
import { cn } from "@/lib/utils"

SyntaxHighlighter.registerLanguage("latex", latex)

interface ConsoleEditorProps {
  value: string
  onChange: (value: string) => void
  language?: string
  placeholder?: string
  readOnly?: boolean
  className?: string
}

export function ConsoleEditor({
  value,
  onChange,
  language = "latex",
  placeholder = "Waiting for output...",
  readOnly = false,
  className,
}: ConsoleEditorProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const backdropRef = useRef<HTMLDivElement>(null)

  const syncScroll = useCallback(() => {
    if (textareaRef.current && backdropRef.current) {
      backdropRef.current.scrollTop = textareaRef.current.scrollTop
      backdropRef.current.scrollLeft = textareaRef.current.scrollLeft
    }
  }, [])

  return (
    <div className={cn("relative flex-1 min-h-0", className)}>
      <div
        ref={backdropRef}
        className="absolute inset-0 overflow-hidden pointer-events-none"
        aria-hidden
      >
        <SyntaxHighlighter
          language={language}
          style={atomOneDark}
          customStyle={{
            margin: 0,
            padding: "1rem",
            fontSize: "0.75rem",
            lineHeight: "1.625",
            fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
            background: "transparent",
            minHeight: "100%",
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
            overflowWrap: "break-word",
          }}
        >
          {value || " "}
        </SyntaxHighlighter>
      </div>
      <textarea
        ref={textareaRef}
        className="relative w-full h-full resize-none bg-transparent p-4 text-xs font-mono leading-relaxed outline-none text-transparent caret-zinc-300 placeholder:text-zinc-600"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onScroll={syncScroll}
        placeholder={placeholder}
        spellCheck={false}
        readOnly={readOnly}
      />
    </div>
  )
}
