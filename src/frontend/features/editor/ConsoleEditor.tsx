"use client"

import { useCallback, useRef } from "react"
import { cn, isModKey } from "@/lib/utils"
import { useHighlightedHtml } from "@/components/content/useHighlightedHtml"

export const LINE_HEIGHT_PX = 12 * 1.625  // 0.75rem × 1.625
export const PADDING_PX = 16              // 1rem — matches p-4

const MONO_FONT = "var(--font-mono), 'JetBrains Mono', ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"

/** A Ctrl/Cmd+I selection: its text and offsets, and the pixel offset just below its last line. */
export interface InlineEditRequest {
  text: string
  start: number
  end: number
  splitPx: number
}

interface ConsoleEditorProps {
  value: string
  onChange: (value: string) => void
  language?: string
  placeholder?: string
  readOnly?: boolean
  className?: string
  startLineNumber?: number
  /** Fired on Ctrl/Cmd+I with a non-empty selection. */
  onInlineEdit?: (request: InlineEditRequest) => void
}

/** Code editor made of a transparent textarea over a Shiki-highlighted backdrop, with a line-number gutter. */
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

  const lineCount   = value.split("\n").length
  const gutterWidth = Math.max(2, String(lineCount).length) * 8 + 16  // px

  const highlightedHtml = useHighlightedHtml(value, language)

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
      {/* Gutter: pinned to the left, scrolls with the textarea */}
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
          if (isModKey(e) && e.key === "i") {
            const ta = textareaRef.current
            if (ta && ta.selectionStart !== ta.selectionEnd && onInlineEdit) {
              e.preventDefault()
              // Pixel offset just below the selection's last line, where the inline edit panel is inserted
              const endLine = ta.value.substring(0, ta.selectionEnd).split("\n").length - 1
              const splitPx = PADDING_PX + (endLine + 1) * LINE_HEIGHT_PX
              onInlineEdit({ text: ta.value.substring(ta.selectionStart, ta.selectionEnd), start: ta.selectionStart, end: ta.selectionEnd, splitPx })
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
