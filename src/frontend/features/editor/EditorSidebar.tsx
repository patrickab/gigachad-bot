"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { ArrowUp, Check, Square, X } from "lucide-react"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"
import { AssistantChat, createFlush, runStream, useAssistantChat } from "@/components/content/AssistantChat"
import { SidePanelFrame } from "@/components/ui/SidePanel"

const SYS_NOTE_EDIT = (doc: string) =>
  `You are an AI writing assistant helping edit a markdown document. Here is the current document:\n\n<document>\n${doc}\n</document>\n\nWhen asked to modify the document, respond with ONLY the complete modified document. No code fences, no explanations — just the raw markdown.`

const SYS_INLINE_EDIT = (sel: string) =>
  `You are an AI writing assistant. The user selected this text:\n\n<selection>\n${sel}\n</selection>\n\nRespond with ONLY the replacement text. No code fences, no explanations.`

interface EditorSidebarProps {
  content: string
  model: string
  width: number
  onWidthChange: (w: number) => void
  onApply: (newContent: string) => void
}

/** Chat sidebar for AI edits of the document. Apply replaces the whole document with a reply. */
export function EditorSidebar({ content, model, width, onWidthChange, onApply }: EditorSidebarProps) {
  const chat = useAssistantChat(model, SYS_NOTE_EDIT(content))
  return (
    <SidePanelFrame width={width} onWidthChange={onWidthChange} className="border-l border-divider bg-paper">
      <AssistantChat
        chat={chat}
        inputLabel="Edit instruction"
        placeholder="Edit instruction…"
        renderActions={(reply) => (
          <button onClick={() => onApply(reply)} className="flex items-center gap-1 text-[10px] text-ink-subtle hover:text-ink transition-colors">
            <Check className="h-3 w-3" /> Apply
          </button>
        )}
      />
    </SidePanelFrame>
  )
}

interface InlineEditPanelProps {
  selectedText: string
  model: string
  onApply: (replacement: string) => void
  onClose: () => void
}

/** Ctrl+I panel: streams a replacement for `selectedText` and applies it on confirm. */
export function InlineEditPanel({ selectedText, model, onApply, onClose }: InlineEditPanelProps) {
  const [input, setInput] = useState("")
  const [response, setResponse] = useState("")
  const [isStreaming, setIsStreaming] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const abortRef = useRef<(() => void) | null>(null)

  useEffect(() => { inputRef.current?.focus() }, [])

  // Capture phase so Escape closes this panel before the editor's own handlers see it
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); abortRef.current?.(); onClose() }
    }
    document.addEventListener("keydown", handler, true)
    return () => document.removeEventListener("keydown", handler, true)
  }, [onClose])

  const handleSend = useCallback(async () => {
    const trimmed = input.trim()
    if (!trimmed || isStreaming) return

    setIsStreaming(true)
    setResponse("")
    // Plain object so the flush closure reads the latest text
    const buf = { text: "" }
    const { schedule, final: flush } = createFlush(() => setResponse(buf.text))

    try {
      await runStream(abortRef, {
        model, chat_id: crypto.randomUUID(), user_msg: trimmed,
        system_prompt: SYS_INLINE_EDIT(selectedText), messages: [], img_paths: [],
      }, (t) => { buf.text += t; schedule() })
    } finally {
      flush()
      setIsStreaming(false)
      abortRef.current = null
    }
  }, [input, isStreaming, selectedText, model])

  return (
    <div className="shrink-0 bg-surface border-y border-divider overflow-hidden">
      {response && (
        <div className="px-4 py-2 max-h-64 overflow-y-auto border-b border-divider/50">
          <div className="text-[11px] leading-relaxed">
            <LaTeXMarkdown content={response} />
          </div>
        </div>
      )}
      <div className="flex items-center gap-2 px-3 py-1.5">
        <input
          ref={inputRef}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => { if (e.key === "Enter") { e.preventDefault(); handleSend() } }}
          placeholder="Describe the edit…"
          className="flex-1 bg-transparent text-xs text-ink placeholder:text-ink-faint outline-none"
        />
        {isStreaming ? (
          <button aria-label="Stop assistant" onClick={() => { abortRef.current?.(); setIsStreaming(false) }} className="p-1 text-ink-subtle hover:text-ink shrink-0">
            <Square className="h-3 w-3" />
          </button>
        ) : (
          <button aria-label="Send to assistant" onClick={handleSend} disabled={!input.trim()} className="p-1 text-ink-subtle hover:text-ink disabled:opacity-30 shrink-0">
            <ArrowUp className="h-3 w-3" />
          </button>
        )}
        <button aria-label="Close assistant" onClick={() => { abortRef.current?.(); onClose() }} className="p-1 text-ink-subtle hover:text-ink shrink-0">
          <X className="h-3 w-3" />
        </button>
      </div>
      {response && !isStreaming && (
        <div className="flex items-center justify-end gap-3 px-3 py-1 border-t border-divider/50">
          <button onClick={onClose} className="text-[10px] text-ink-subtle hover:text-ink transition-colors">Close</button>
          <button onClick={() => onApply(response)} className="flex items-center gap-1 text-[10px] text-ink-subtle hover:text-ink transition-colors">
            <Check className="h-3 w-3" /> Apply
          </button>
        </div>
      )}
    </div>
  )
}
