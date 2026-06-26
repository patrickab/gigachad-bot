"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { ArrowUp, Check, Square, X } from "lucide-react"
import { motion, useMotionValue, useTransform, animate } from "framer-motion"
import { createChatStream } from "@/lib/api"
import { LaTeXMarkdown } from "./LaTeXMarkdown"
import { MIN_SIDEBAR_WIDTH, MAX_SIDEBAR_WIDTH } from "./ChatSidebar"

interface NoteMessage {
  role: "user" | "assistant"
  content: string
}

const FLUSH_MS = 60

const SYS_NOTE_EDIT = (doc: string) =>
  `You are an AI writing assistant helping edit a markdown document. Here is the current document:\n\n<document>\n${doc}\n</document>\n\nWhen asked to modify the document, respond with ONLY the complete modified document. No code fences, no explanations — just the raw markdown.`

const SYS_INLINE_EDIT = (sel: string) =>
  `You are an AI writing assistant. The user selected this text:\n\n<selection>\n${sel}\n</selection>\n\nRespond with ONLY the replacement text. No code fences, no explanations.`

async function runStream(
  abortRef: { current: (() => void) | null },
  args: Parameters<typeof createChatStream>[0],
  onToken: (t: string) => void
) {
  try {
    const stream = createChatStream(args)
    abortRef.current = stream.abort
    for await (const ev of stream) {
      if (ev.event === "token") onToken(ev.data)
      else if (ev.event === "done" || ev.event === "error") break
    }
  } catch (e) {
    if ((e as Error).name !== "AbortError") throw e
  }
}

function createFlush(onFlush: () => void) {
  let last = 0
  let timer: ReturnType<typeof setTimeout> | null = null
  const schedule = () => {
    if (timer) return
    const delay = Math.max(0, FLUSH_MS - (performance.now() - last))
    if (delay <= 0) { last = performance.now(); onFlush(); return }
    timer = setTimeout(() => { timer = null; last = performance.now(); onFlush() }, delay)
  }
  const final = () => { if (timer) clearTimeout(timer); onFlush() }
  return { schedule, final }
}

// ─── Sidebar ───

interface EditorSidebarProps {
  content: string
  model: string
  width: number
  onWidthChange: (w: number) => void
  onApply: (newContent: string) => void
}

export function EditorSidebar({ content, model, width, onWidthChange, onApply }: EditorSidebarProps) {
  const [messages, setMessages] = useState<NoteMessage[]>([])
  const [input, setInput] = useState("")
  const [isStreaming, setIsStreaming] = useState(false)
  const contentRef = useRef(content)
  contentRef.current = content
  const scrollRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<(() => void) | null>(null)
  const dragging = useRef(false)
  const motionWidth = useMotionValue(width)

  useEffect(() => {
    if (dragging.current) return
    const controls = animate(motionWidth, width, { type: "spring", stiffness: 400, damping: 40, mass: 0.6 })
    return controls.stop
  }, [width, motionWidth])

  const widthStyle = useTransform(motionWidth, (v) => `${v}px`)

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    dragging.current = true
    const startX = e.clientX
    const startWidth = motionWidth.get()
    const onMouseMove = (ev: MouseEvent) => {
      if (!dragging.current) return
      motionWidth.set(Math.min(MAX_SIDEBAR_WIDTH, Math.max(MIN_SIDEBAR_WIDTH, startWidth + (startX - ev.clientX))))
    }
    const onMouseUp = () => {
      if (!dragging.current) return
      dragging.current = false
      const final = Math.min(MAX_SIDEBAR_WIDTH, Math.max(MIN_SIDEBAR_WIDTH, motionWidth.get()))
      motionWidth.set(final)
      onWidthChange(final)
      document.removeEventListener("mousemove", onMouseMove)
      document.removeEventListener("mouseup", onMouseUp)
    }
    document.addEventListener("mousemove", onMouseMove)
    document.addEventListener("mouseup", onMouseUp)
  }, [motionWidth, onWidthChange])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" })
  }, [messages])

  const handleSend = useCallback(async () => {
    const trimmed = input.trim()
    if (!trimmed || isStreaming) return

    const userMsg: NoteMessage = { role: "user", content: trimmed }
    const assistantMsg: NoteMessage = { role: "assistant", content: "" }
    setMessages(prev => [...prev, userMsg, assistantMsg])
    setInput("")
    setIsStreaming(true)

    const history = messages.map(m => ({ role: m.role, content: m.content }))
    const { schedule, final: flush } = createFlush(() => {
      setMessages(prev => {
        const copy = [...prev]
        copy[copy.length - 1] = { ...assistantMsg }
        return copy
      })
    })

    try {
      await runStream(abortRef, {
        model, chat_id: crypto.randomUUID(), user_msg: trimmed,
        system_prompt: SYS_NOTE_EDIT(contentRef.current), messages: history, img_paths: [],
      }, (t) => { assistantMsg.content += t; schedule() })
    } finally {
      flush()
      setIsStreaming(false)
      abortRef.current = null
    }
  }, [input, isStreaming, messages, model])

  return (
    <motion.aside
      initial={false}
      style={{ width: widthStyle }}
      className="shrink-0 border-l border-divider bg-paper flex flex-col overflow-hidden relative"
    >
      <div
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize sidebar"
        className="absolute left-0 top-0 bottom-0 w-1 cursor-col-resize hover:bg-surface-elevated active:bg-surface-elevated focus-visible:bg-surface-elevated transition-colors z-10"
        onMouseDown={handleMouseDown}
      />
      <div ref={scrollRef} className="flex-1 overflow-y-auto min-h-0 p-3 space-y-3">
        {messages.map((msg, i) => (
          <div key={i}>
            {msg.role === "user" ? (
              <div className="bg-surface rounded-md px-3 py-2 text-xs text-ink-muted">{msg.content}</div>
            ) : (
              <div className="space-y-1.5 pt-1">
                <div className="text-[11px] leading-relaxed overflow-hidden">
                  <LaTeXMarkdown content={msg.content || "…"} />
                </div>
                {msg.content && !(isStreaming && i === messages.length - 1) && (
                  <button onClick={() => onApply(msg.content)} className="flex items-center gap-1 text-[10px] text-ink-subtle hover:text-ink transition-colors">
                    <Check className="h-3 w-3" /> Apply
                  </button>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
      <div className="shrink-0 border-t border-divider/50 p-2">
        <div className="flex items-end gap-1.5 bg-surface rounded-md px-2 py-1.5">
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend() } }}
            rows={1}
            placeholder="Edit instruction…"
            className="flex-1 resize-none bg-transparent text-xs text-ink placeholder:text-ink-faint outline-none max-h-20 overflow-y-auto"
          />
          {isStreaming ? (
            <button onClick={() => { abortRef.current?.(); setIsStreaming(false) }} className="p-1 rounded text-ink-subtle hover:text-ink transition-colors shrink-0">
              <Square className="h-3.5 w-3.5" />
            </button>
          ) : (
            <button onClick={handleSend} disabled={!input.trim()} className="p-1 rounded text-ink-subtle hover:text-ink disabled:opacity-30 transition-colors shrink-0">
              <ArrowUp className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      </div>
    </motion.aside>
  )
}

// ─── Inline Edit Panel (Ctrl+I) ───

interface InlineEditPanelProps {
  selectedText: string
  model: string
  onApply: (replacement: string) => void
  onClose: () => void
}

export function InlineEditPanel({ selectedText, model, onApply, onClose }: InlineEditPanelProps) {
  const [input, setInput] = useState("")
  const [response, setResponse] = useState("")
  const [isStreaming, setIsStreaming] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const abortRef = useRef<(() => void) | null>(null)

  useEffect(() => { inputRef.current?.focus() }, [])

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
          <button onClick={() => { abortRef.current?.(); setIsStreaming(false) }} className="p-1 text-ink-subtle hover:text-ink shrink-0">
            <Square className="h-3 w-3" />
          </button>
        ) : (
          <button onClick={handleSend} disabled={!input.trim()} className="p-1 text-ink-subtle hover:text-ink disabled:opacity-30 shrink-0">
            <ArrowUp className="h-3 w-3" />
          </button>
        )}
        <button onClick={() => { abortRef.current?.(); onClose() }} className="p-1 text-ink-subtle hover:text-ink shrink-0">
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
