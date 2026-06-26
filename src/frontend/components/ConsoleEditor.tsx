"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { cn } from "@/lib/utils"
import { highlightCode } from "@/lib/markdown-syntax-highlighting"

const LINE_HEIGHT_PX = 12 * 1.625  // 0.75rem × 1.625
const PADDING_PX = 16              // 1rem — matches p-4

const MONO_FONT = "var(--font-mono), 'JetBrains Mono', ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"

interface ConsoleEditorProps {
  value: string
  onChange: (value: string) => void
  language?: string
  placeholder?: string
  readOnly?: boolean
  className?: string
  startLineNumber?: number
  onInlineEdit?: (selectedText: string, selectionStart: number, selectionEnd: number, splitPx: number) => void
}

export function ConsoleEditor({
  value,
  onChange,
  language = "latex",
  placeholder = "Waiting for output...",
  readOnly = false,
  className,
  startLineNumber = 1,
  onInlineEdit,
}: ConsoleEditorProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const backdropRef = useRef<HTMLDivElement>(null)
  const gutterRef  = useRef<HTMLDivElement>(null)
  const [highlightedHtml, setHighlightedHtml] = useState("")

  const lineCount   = value.split("\n").length
  const gutterWidth = Math.max(2, String(lineCount).length) * 8 + 16  // px

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
    const ta = textareaRef.current
    const bd = backdropRef.current
    const gt = gutterRef.current
    if (ta && bd) { bd.scrollTop = ta.scrollTop; bd.scrollLeft = ta.scrollLeft }
    if (ta && gt) { gt.scrollTop = ta.scrollTop }
  }, [])

  const leftPad = `calc(${gutterWidth}px + ${PADDING_PX}px)`

  return (
    <div className={cn("relative flex-1 min-h-0", className)}>
      {/* Gutter — absolutely pinned to left, scrolls with textarea */}
      <div
        ref={gutterRef}
        className="absolute left-0 top-0 bottom-0 z-10 overflow-hidden pointer-events-none select-none text-right text-ink-faint border-r border-divider/30 bg-paper"
        style={{ width: gutterWidth, fontSize: "0.75rem", lineHeight: "1.625", fontFamily: MONO_FONT, paddingTop: PADDING_PX, paddingBottom: PADDING_PX }}
      >
        {Array.from({ length: lineCount }, (_, i) => (
          <div key={i} className="pr-2">{i + startLineNumber}</div>
        ))}
      </div>
      {/* Highlighted backdrop */}
      <div
        ref={backdropRef}
        className="absolute inset-0 overflow-hidden pointer-events-none select-none"
        aria-hidden
      >
        {highlightedHtml ? (
          <div
            className="shiki-wrapper min-h-full"
            style={{ fontSize: "0.75rem", lineHeight: "1.625", fontFamily: MONO_FONT, whiteSpace: "pre-wrap", overflowWrap: "break-word", paddingTop: PADDING_PX, paddingRight: PADDING_PX, paddingBottom: PADDING_PX, paddingLeft: leftPad }}
            dangerouslySetInnerHTML={{ __html: highlightedHtml }}
          />
        ) : (
          <div className="p-4 text-xs text-ink-faint" style={{ fontFamily: MONO_FONT, paddingLeft: leftPad }}>{value}</div>
        )}
      </div>
      {/* Transparent textarea on top */}
      <textarea
        ref={textareaRef}
        className="console-textarea relative h-full w-full resize-none overflow-auto bg-transparent text-transparent caret-ink outline-none placeholder:text-ink-faint whitespace-pre-wrap break-words"
        style={{ fontSize: "0.75rem", lineHeight: "1.625", fontFamily: MONO_FONT, paddingTop: PADDING_PX, paddingRight: PADDING_PX, paddingBottom: PADDING_PX, paddingLeft: leftPad }}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onScroll={syncScroll}
        onKeyDown={(e) => {
          if ((e.ctrlKey || e.metaKey) && e.key === "i") {
            const ta = textareaRef.current
            if (ta && ta.selectionStart !== ta.selectionEnd && onInlineEdit) {
              e.preventDefault()
              const endLine = ta.value.substring(0, ta.selectionEnd).split("\n").length - 1
              const splitPx = PADDING_PX + (endLine + 1) * LINE_HEIGHT_PX
              onInlineEdit(ta.value.substring(ta.selectionStart, ta.selectionEnd), ta.selectionStart, ta.selectionEnd, splitPx)
            }
          }
        }}
        placeholder={placeholder}
        spellCheck={false}
        readOnly={readOnly}
      />
    </div>
  )
}
