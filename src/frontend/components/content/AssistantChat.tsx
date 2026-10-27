"use client"

import { useCallback, useEffect, useRef, useState, type ReactNode } from "react"
import { ArrowUp, Square } from "lucide-react"
import { createChatStream } from "@/lib/backend/api"
import type { ChatRequest } from "@/lib/domain/types"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"
import { Composer } from "@/components/ui/Composer"

export interface AssistantMessage {
  role: "user" | "assistant"
  content: string
}

const FLUSH_MS = 60

/** Pump a chat stream into `onToken`, exposing its abort via `abortRef`. Abort is swallowed silently. */
async function runStream(
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

/** Request for one stateless reply: `messages` is the history before `userMsg`. */
function replyRequest(model: string, userMsg: string, systemPrompt: string, messages: AssistantMessage[]): ChatRequest {
  return { model, chat_id: crypto.randomUUID(), user_msg: userMsg, system_prompt: systemPrompt, messages, img_paths: [] }
}

/** The text a failed stream appends to its reply. */
function errorLine(error: unknown) {
  return `\n\n**Error:** ${(error as Error).message}`
}

/** One reply streaming at a time, aborted when the owner unmounts. */
function useReplyStream() {
  const [isStreaming, setIsStreaming] = useState(false)
  const abortRef = useRef<(() => void) | null>(null)

  // Abort a running reply when the owner unmounts
  useEffect(() => () => abortRef.current?.(), [])

  /** Stream `request` into `onToken`, calling `show` at most once per FLUSH_MS and once at the end. `onError` gets a failed stream, rethrow to reject. */
  const stream = useCallback(async (request: ChatRequest, onToken: (token: string) => void, show: () => void, onError: (error: unknown) => void) => {
    setIsStreaming(true)
    const { schedule, final: flush } = createFlush(show)
    try {
      await runStream(abortRef, request, (token) => { onToken(token); schedule() })
    } catch (error) {
      onError(error)
    } finally {
      flush()
      setIsStreaming(false)
      abortRef.current = null
    }
  }, [])

  const abort = useCallback(() => abortRef.current?.(), [])
  const stop = useCallback(() => { abortRef.current?.(); setIsStreaming(false) }, [])

  return { isStreaming, stream, abort, stop }
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
  const { isStreaming, stream, stop } = useReplyStream()

  const send = useCallback(async (text: string) => {
    const trimmed = text.trim()
    if (!trimmed || isStreaming) return

    const userMsg: AssistantMessage = { role: "user", content: trimmed }
    const assistantMsg: AssistantMessage = { role: "assistant", content: "" }
    setMessages(prev => [...prev, userMsg, assistantMsg])

    // Snapshot before the new messages land. `assistantMsg` is mutated per token and copied on flush
    const history = messages.map(m => ({ role: m.role, content: m.content }))
    await stream(replyRequest(model, trimmed, systemPrompt, history), (t) => { assistantMsg.content += t }, () => {
      setMessages(prev => {
        const copy = [...prev]
        copy[copy.length - 1] = { ...assistantMsg }
        return copy
      })
    }, (e) => { assistantMsg.content += errorLine(e) })
  }, [isStreaming, messages, model, systemPrompt, stream])

  return { messages, isStreaming, send, stop }
}

/** One streamed reply to a single question. `appendErrors` shows a failed stream as an error line, otherwise `ask` rejects. */
export function useOneShotStream(model: string, { appendErrors }: { appendErrors: boolean }) {
  const [text, setText] = useState("")
  const { isStreaming, stream, abort, stop } = useReplyStream()

  const ask = useCallback(async (question: string, systemPrompt: string) => {
    const trimmed = question.trim()
    if (!trimmed || isStreaming) return
    setText("")
    // Plain object so the flush closure reads the latest text
    const buffer = { text: "" }
    await stream(replyRequest(model, trimmed, systemPrompt, []), (token) => { buffer.text += token }, () => setText(buffer.text), (error) => {
      if (!appendErrors) throw error
      buffer.text += errorLine(error)
    })
  }, [isStreaming, model, appendErrors, stream])

  return { text, isStreaming, ask, abort, stop }
}

/** Stop button while streaming, otherwise the send button. */
export function SendStopButton({ isStreaming, onStop, onSend, disabled, sendLabel = "Send to assistant", className, iconClassName }: {
  isStreaming: boolean
  onStop: () => void
  onSend: () => void
  disabled: boolean
  sendLabel?: string
  className: string
  iconClassName: string
}) {
  return isStreaming ? (
    <button aria-label="Stop assistant" onClick={onStop} className={className}>
      <Square className={iconClassName} />
    </button>
  ) : (
    <button aria-label={sendLabel} onClick={onSend} disabled={disabled} className={`${className} disabled:opacity-30`}>
      <ArrowUp className={iconClassName} />
    </button>
  )
}

/** Message list and composer of an assistant chat. `renderActions` adds controls under finished replies. `surfaceClassName` lifts user messages off their host. */
export function AssistantChat({ chat, inputLabel, placeholder = "Message", disabled = false, surfaceClassName = "bg-surface", renderActions }: {
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
      <div className="shrink-0 p-3 pt-0">
        <Composer
          value={input}
          onValueChange={setInput}
          onSubmit={handleSend}
          onStop={stop}
          label={inputLabel}
          placeholder={placeholder}
          disabled={disabled}
          isStreaming={isStreaming}
        />
      </div>
    </>
  )
}
