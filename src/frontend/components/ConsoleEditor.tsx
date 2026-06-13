"use client"

import type { CSSProperties } from "react"
import { cn } from "@/lib/utils"

interface ConsoleEditorProps {
  value: string
  onChange: (value: string) => void
  language?: string
  placeholder?: string
  readOnly?: boolean
  className?: string
}

const MONO_FONT = "var(--font-mono), 'JetBrains Mono', ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, 'Liberation Mono', monospace"
const EDITOR_METRICS: CSSProperties = {
  fontFamily: MONO_FONT,
  fontSize: "0.75rem",
  fontWeight: 400,
  fontStyle: "normal",
  fontVariantLigatures: "none",
  lineHeight: 1.625,
  letterSpacing: "normal",
  tabSize: 2,
  whiteSpace: "pre-wrap",
  overflowWrap: "break-word",
  wordBreak: "break-word",
}

export function ConsoleEditor({
  value,
  onChange,
  placeholder = "Waiting for output...",
  readOnly = false,
  className,
}: ConsoleEditorProps) {
  return (
    <div className={cn("relative flex-1 min-h-0", className)}>
      <textarea
        className="relative h-full w-full resize-none overflow-auto bg-transparent p-4 text-ink outline-none placeholder:text-ink-faint whitespace-pre-wrap break-words"
        style={EDITOR_METRICS}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        spellCheck={false}
        readOnly={readOnly}
      />
    </div>
  )
}
