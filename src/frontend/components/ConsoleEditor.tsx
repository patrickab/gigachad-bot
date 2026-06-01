"use client"

import { useCallback, useMemo, useRef } from "react"
import { highlightCode } from "@/lib/markdown-syntax-highlighting"
import { cn } from "@/lib/utils"

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

  const highlightedHtml = useMemo(
    () => highlightCode(value || " ", language),
    [value, language],
  )

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
        className="absolute inset-0 overflow-hidden pointer-events-none [&_pre]:!bg-transparent [&_pre]:!p-4 [&_pre]:!m-0 [&_pre]:!text-[0.75rem] [&_pre]:!leading-[1.625] [&_pre]:!font-mono [&_pre]:!whitespace-pre-wrap [&_pre]:!break-words [&_pre]:!min-h-full"
        aria-hidden
        dangerouslySetInnerHTML={{ __html: highlightedHtml }}
      />
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
