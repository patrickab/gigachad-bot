"use client"

import { useCallback, useState, type ReactNode } from "react"
import { cn } from "@/lib/utils"
import { ConsoleEditor, LINE_HEIGHT_PX, PADDING_PX, type InlineEditRequest } from "./ConsoleEditor"
import { InlineEditPanel } from "./EditorSidebar"

interface AssistedConsoleEditorProps {
  value: string
  onChange: (value: string) => void
  language: string
  /** Enables Ctrl+I inline AI edits; omitted means plain editing. */
  model?: string
  placeholder?: string
  className?: string
  /** Rendered inside the editor frame, before the editor (e.g. floating pills). */
  children?: ReactNode
}

/** `ConsoleEditor` plus Ctrl+I inline AI edit: the value splits after the selection's line around an edit panel. */
export function AssistedConsoleEditor({ value, onChange, language, model, placeholder, className, children }: AssistedConsoleEditorProps) {
  const [inlineEdit, setInlineEdit] = useState<InlineEditRequest | null>(null)

  const handleInlineApply = useCallback((replacement: string) => {
    if (!inlineEdit) return
    onChange(value.substring(0, inlineEdit.start) + replacement + value.substring(inlineEdit.end))
    setInlineEdit(null)
  }, [inlineEdit, value, onChange])

  const split = inlineEdit && model ? splitAfterLine(value, inlineEdit.end) : null

  // While splitting, the frame is a plain scroll viewport so the whole value scrolls with the panel inline
  return (
    <div className={cn("relative flex-1 min-h-0", split ? "overflow-y-auto" : "flex flex-col", className)}>
      {children}
      {inlineEdit && model && split ? (
        <>
          <div style={{ height: inlineEdit.splitPx }} className="flex flex-col shrink-0 overflow-hidden">
            <ConsoleEditor value={split.top} onChange={() => {}} language={language} readOnly />
          </div>
          <InlineEditPanel
            selectedText={inlineEdit.text}
            model={model}
            onApply={handleInlineApply}
            onClose={() => setInlineEdit(null)}
          />
          {/* Pixel height of the bottom slice so the scroll viewport knows its true extent */}
          <div style={{ height: PADDING_PX + split.bottom.split("\n").length * LINE_HEIGHT_PX + PADDING_PX }} className="flex flex-col shrink-0">
            <ConsoleEditor value={split.bottom} onChange={() => {}} language={language} startLineNumber={split.top.split("\n").length} readOnly />
          </div>
        </>
      ) : (
        <ConsoleEditor value={value} onChange={onChange} language={language} placeholder={placeholder} onInlineEdit={model ? setInlineEdit : undefined} />
      )}
    </div>
  )
}

/** Split `text` right after the newline ending the line that contains `offset`. */
function splitAfterLine(text: string, offset: number): { top: string; bottom: string } {
  const newline = text.indexOf("\n", offset)
  const at = newline === -1 ? text.length : newline + 1
  return { top: text.substring(0, at), bottom: text.substring(at) }
}
