"use client"

import { useCallback, useEffect, useRef, useState, type ReactNode } from "react"
import { ArrowUp, Square } from "lucide-react"
import { createChatStream } from "@/lib/backend/api"
import type { ChatRequest } from "@/lib/domain/types"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"

export interface AssistantMessage {
  role: "user" | "assistant"
  content: string
}

const FLUSH_MS = 60

/** Pump a chat stream into `onToken`, exposing its abort via `abortRef`. Abort is swallowed silently. */
export async function runStream(
  abortRef: { current: (() => void) | null },
  args: ChatRequest,
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

/** Throttle flushes to one per FLUSH_MS. `final` flushes immediately and cancels the pending timer. */
export function createFlush(onFlush: () => void) {
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

export interface AssistantChatState {
  messages: AssistantMessage[]
  isStreaming: boolean
  send: (text: string) => void
  stop: () => void
}

/** Stateless-backend chat: the whole history and `systemPrompt` go with every turn. */
export function useAssistantChat(model: string, systemPrompt: string): AssistantChatState {
  const [messages, setMessages] = useState<AssistantMessage[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const abortRef = useRef<(() => void) | null>(null)

  // Abort a running reply when the owner unmounts
  useEffect(() => () => abortRef.current?.(), [])

  const send = useCallback(async (text: string) => {
    const trimmed = text.trim()
    if (!trimmed || isStreaming) return

    const userMsg: AssistantMessage = { role: "user", content: trimmed }
    const assistantMsg: AssistantMessage = { role: "assistant", content: "" }
    setMessages(prev => [...prev, userMsg, assistantMsg])
    setIsStreaming(true)

    // Snapshot before the new messages land. `assistantMsg` is mutated per token and copied on flush
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
        system_prompt: systemPrompt, messages: history, img_paths: [],
      }, (t) => { assistantMsg.content += t; schedule() })
    } catch (e) {
      assistantMsg.content += `\n\n**Error:** ${(e as Error).message}`
    } finally {
      flush()
      setIsStreaming(false)
      abortRef.current = null
    }
  }, [isStreaming, messages, model, systemPrompt])

  const stop = useCallback(() => { abortRef.current?.(); setIsStreaming(false) }, [])

  return { messages, isStreaming, send, stop }
}

/** Message list and composer of an assistant chat. `renderActions` adds controls under finished replies. `surfaceClassName` lifts user messages and the composer off their host. */
export function AssistantChat({ chat, inputLabel, placeholder, disabled = false, surfaceClassName = "bg-surface", renderActions }: {
  chat: AssistantChatState
  inputLabel: string
  placeholder?: string
  disabled?: boolean
  surfaceClassName?: string
  renderActions?: (content: string) => ReactNode
}) {
  const { messages, isStreaming, send, stop } = chat
  const [input, setInput] = useState("")
  const scrollRef = useRef<HTMLDivElement>(null)
  // Keep the newest message in view
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" })
  }, [messages])

  const handleSend = () => {
    if (!input.trim() || isStreaming || disabled) return
    send(input)
    setInput("")
  }

  return (
    <>
      <div ref={scrollRef} className="flex-1 overflow-y-auto min-h-0 p-3 space-y-3">
        {messages.map((msg, i) => (
          <div key={i}>
            {msg.role === "user" ? (
              <div className={`rounded-md px-3 py-2 text-xs text-ink-muted ${surfaceClassName}`}>{msg.content}</div>
            ) : (
              <div className="space-y-1.5 pt-1">
                <div className="text-[11px] leading-relaxed overflow-hidden">
                  <LaTeXMarkdown content={msg.content || "…"} />
                </div>
                {renderActions && msg.content && !(isStreaming && i === messages.length - 1) && renderActions(msg.content)}
              </div>
            )}
          </div>
        ))}
      </div>
      <div className="shrink-0 border-t border-divider/50 p-2">
        <div className={`flex items-end gap-1.5 rounded-md px-2 py-1.5 ${surfaceClassName}`}>
          <textarea
            aria-label={inputLabel}
            value={input}
            disabled={disabled}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend() } }}
            rows={1}
            placeholder={placeholder}
            className="flex-1 resize-none bg-transparent text-xs text-ink placeholder:text-ink-faint outline-none max-h-20 overflow-y-auto disabled:opacity-50"
          />
          {isStreaming ? (
            <button aria-label="Stop assistant" onClick={stop} className="p-1 rounded text-ink-subtle hover:text-ink transition-colors shrink-0">
              <Square className="h-3.5 w-3.5" />
            </button>
          ) : (
            <button aria-label="Send to assistant" onClick={handleSend} disabled={!input.trim() || disabled} className="p-1 rounded text-ink-subtle hover:text-ink disabled:opacity-30 transition-colors shrink-0">
              <ArrowUp className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      </div>
    </>
  )
}
