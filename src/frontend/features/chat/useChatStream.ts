"use client"

import { useCallback, useLayoutEffect, useRef, useState } from "react"
import { createChatStream, currentConnectionPool, leaveConnectionPool } from "@/lib/backend/api"
import { deactivateSentImages } from "@/features/chat/attachments"
import { createTokenPacer } from "@/lib/backend/tokenPacer"
import {
  applyChatStreamEvent,
  createMessagePublisher,
  decodeChatStreamEvent,
  settleRunningToolCalls,
} from "@/lib/backend/streaming"
import { toolPayload } from "@/features/chat/toolCalls"
import type { ChatRequest, Message, Usage } from "@/lib/domain/types"
import { errorMessage } from "@/lib/utils"

export interface ChatStreamCompletion {
  messages: Message[]
  usage: Usage
}

export interface UseChatStreamReturn {
  messages: Message[]
  isStreaming: boolean
  /** `skipAddMessages` streams into the existing trailing assistant message instead of appending a user/assistant pair. */
  send: (req: ChatRequest, skipAddMessages?: boolean) => Promise<ChatStreamCompletion | null>
  cancel: () => void
  /** Take the pair starting at `user` out of the chat, with every later pair when `through` is "end". Returns the new list, null when `user` is not in the chat. */
  removePairs: (user: Message, through: "pair" | "end") => Message[] | null
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>
  /** Update the latest messages so a `send` right after builds its history from them, without waiting for a render. Returns the result. */
  replaceMessages: (update: (current: Message[]) => Message[]) => Message[]
  totalUsage: Usage
  setTotalUsage: React.Dispatch<React.SetStateAction<Usage>>
}

/** Build the API history, prepending hidden attachment text to user turns and appending mindmap output to assistant turns. */
function buildHistory(msgs: Message[]): { role: string; content: string }[] {
  return msgs
    .filter(m => m.role === "user" || m.role === "assistant")
    .map(m => {
      let content = m.content
      if (m.role === "user" && m.hiddenContent) {
        content = m.hiddenContent + "\n\n" + content
      }
      const mindmap = m.tool_calls?.map(toolPayload).find((payload) => payload.family === "mindmap")
      if (mindmap?.family === "mindmap" && mindmap.content) content = [content, mindmap.content].filter(Boolean).join("\n\n")
      return { role: m.role, content }
    })
}

/** `messages` without the pair whose question is `user` (`"pair"`), or without it and every later one (`"end"`); null when `user` is not there. */
export function withoutPairs(messages: Message[], user: Message, through: "pair" | "end"): Message[] | null {
  const index = messages.indexOf(user)
  if (index < 0) return null
  // A pair whose answer is missing is the user message alone
  const end = through === "end" ? messages.length : index + (messages[index + 1]?.role === "assistant" ? 2 : 1)
  return [...messages.slice(0, index), ...messages.slice(end)]
}

/** Own the message list and stream one assistant reply at a time. `send` resolves to null when cancelled, failed, or not completed. */
export function useChatStream(): UseChatStreamReturn {
  const [messages, setMessages] = useState<Message[]>([])
  const [totalUsage, setTotalUsage] = useState<Usage>({ prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 })
  const [isStreaming, setIsStreaming] = useState(false)
  const abortRef = useRef<(() => void) | null>(null)
  // Refs mirror state so callbacks read the latest values without stale closures
  const messagesRef = useRef(messages)
  const totalUsageRef = useRef(totalUsage)
  useLayoutEffect(() => {
    messagesRef.current = messages
    totalUsageRef.current = totalUsage
  }, [messages, totalUsage])

  const streamAssistantReply = useCallback(
    async (req: ChatRequest, history: { role: string; content: string }[], assistantMsg: Message): Promise<ChatStreamCompletion | null> => {
      setIsStreaming(true)

      // Every message this reply put at the end of the list. Another chat opened meanwhile replaces it: the reply then leaves the list alone.
      const mine = new Set([messagesRef.current.at(-1)])
      const publish = createMessagePublisher((update) => setMessages((prev) => {
        const next = typeof update === "function" ? update(prev) : update
        if (next !== prev) mine.add(next.at(-1))
        return next
      }), assistantMsg)
      // Answer text is paced, everything else lands as it arrives.
      const pacer = createTokenPacer((text) => {
        assistantMsg.content += text
        publish()
      })
      // Every exit from the loop below settles the tool cards still marked running.
      let unresolvedReason = "Stream ended before the tool returned"
      let completed = false
      let usage = totalUsageRef.current
      let finalMessages: Message[] | null = null

      try {
        for (let attempt = 0; ; attempt++) {
          let received = false
          const pool = currentConnectionPool()
          try {
            const stream = createChatStream({ ...req, messages: history })
            abortRef.current = stream.abort
            for await (const event of stream) {
              received = true
              const decoded = decodeChatStreamEvent(event)
              if (!decoded) continue
              if (decoded.kind === "usage") {
                usage = decoded.usage
                totalUsageRef.current = usage
                setTotalUsage(usage)
                continue
              }
              if (decoded.kind === "done") {
                completed = true
                await pacer.finish()
                break
              }
              if (decoded.kind === "token") {
                pacer.push(decoded.text)
                continue
              }
              if (decoded.kind === "error") pacer.flush()
              applyChatStreamEvent(assistantMsg, decoded)
              publish()
              // An error event is the turn's last content: append it, then stop reading.
              if (decoded.kind === "error") {
                unresolvedReason = decoded.message
                break
              }
            }
            break
          } catch (e) {
            // A stream that died before any event is sent again, over a fresh connection
            if (received || attempt >= 2 || !(e instanceof TypeError)) throw e
            leaveConnectionPool(pool)
          }
        }
      } catch (e) {
        // An abort keeps what streamed. A transport failure also says so in the reply, or it would show an empty bubble.
        const aborted = e instanceof Error && e.name === "AbortError"
        unresolvedReason = aborted ? "Cancelled" : errorMessage(e, "Stream failed")
        if (!aborted) {
          pacer.flush()
          applyChatStreamEvent(assistantMsg, { kind: "error", message: unresolvedReason })
        }
      } finally {
        pacer.flush()
        settleRunningToolCalls(assistantMsg, unresolvedReason)
        publish()
        const current = messagesRef.current
        if (current.at(-1)?.role === "assistant" && mine.has(current.at(-1))) {
          finalMessages = deactivateSentImages(
            [...current.slice(0, -1), { ...assistantMsg }],
            req.img_paths,
          )
          messagesRef.current = finalMessages
          setMessages(finalMessages)
        }
        setIsStreaming(false)
        abortRef.current = null
      }

      return completed && finalMessages ? { messages: finalMessages, usage } : null
    },
    []
  )

  const send = useCallback(
    async (req: ChatRequest, skipAddMessages = false) => {
      const assistantMsg: Message = { role: "assistant", content: "" }

      if (!skipAddMessages) {
        const next = [
          ...messagesRef.current,
          { role: "user" as const, content: req.user_msg },
          assistantMsg,
        ]
        messagesRef.current = next
        setMessages(next)
      }

      const history = buildHistory(messagesRef.current.slice(0, -2))
      return streamAssistantReply(req, history, assistantMsg)
    },
    [streamAssistantReply]
  )

  const cancel = useCallback(() => {
    abortRef.current?.()
    setIsStreaming(false)
  }, [])

  const replaceMessages = useCallback((update: (current: Message[]) => Message[]) => {
    const next = update(messagesRef.current)
    messagesRef.current = next
    setMessages(next)
    return next
  }, [])

  const removePairs = useCallback((user: Message, through: "pair" | "end") => {
    const next = withoutPairs(messagesRef.current, user, through)
    return next && replaceMessages(() => next)
  }, [replaceMessages])

  return { messages, isStreaming, send, cancel, removePairs, setMessages, replaceMessages, totalUsage, setTotalUsage }
}