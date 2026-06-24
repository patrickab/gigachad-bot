"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { cn } from "@/lib/utils"
import { highlightCode } from "@/lib/markdown-syntax-highlighting"

interface ConsoleEditorProps {
  value: string
  onChange: (value: string) => void
  language?: string
  placeholder?: string
  readOnly?: boolean
  className?: string
}

const MONO_FONT = "var(--font-mono), 'JetBrains Mono', ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"

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
  const [highlightedHtml, setHighlightedHtml] = useState("")

  useEffect(() => {
    let cancelled = false
    highlightCode(value, language).then((html) => {
      if (!cancelled) setHighlightedHtml(html)
    }).catch(() => {
      if (!cancelled) setHighlightedHtml("")
    })
    return () => { cancelled = true }
  }, [value, language])

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
        className="absolute inset-0 overflow-hidden pointer-events-none select-none"
        aria-hidden
      >
        {highlightedHtml ? (
          <div
            className="shiki-wrapper min-h-full p-4"
            style={{ fontSize: "0.75rem", lineHeight: "1.625", fontFamily: MONO_FONT, whiteSpace: "pre-wrap", overflowWrap: "break-word" }}
            dangerouslySetInnerHTML={{ __html: highlightedHtml }}
          />
        ) : (
          <div className="p-4 text-xs text-ink-faint" style={{ fontFamily: MONO_FONT }}>{value}</div>
        )}
      </div>
      <textarea
        ref={textareaRef}
        className="console-textarea relative h-full w-full resize-none overflow-auto bg-transparent p-4 text-transparent caret-ink outline-none placeholder:text-ink-faint whitespace-pre-wrap break-words"
        style={{ fontSize: "0.75rem", lineHeight: "1.625", fontFamily: MONO_FONT }}
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
