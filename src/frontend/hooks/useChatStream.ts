"use client"

import { useCallback, useRef, useState } from "react"
import { createChatStream } from "@/lib/api"
import { deactivateSentImages } from "@/lib/attachments"
import {
  addUsage,
  applyChatStreamEvent,
  createFlushBatcher,
  decodeChatStreamEvent,
  settleRunningToolCalls,
} from "@/lib/streaming"
import type { ChatRequest, Message, Usage } from "@/lib/types"

export interface UseChatStreamReturn {
  messages: Message[]
  isStreaming: boolean
  send: (req: ChatRequest, skipAddMessages?: boolean) => Promise<void>
  regenerateAt: (userIndex: number, req: ChatRequest) => Promise<void>
  cancel: () => void
  deleteMessagePair: (index: number) => void
  addMessagePair: (userContent: string, assistantContent: string) => void
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>
  totalUsage: Usage
  setTotalUsage: React.Dispatch<React.SetStateAction<Usage>>
}

function buildHistory(msgs: Message[]): { role: string; content: string }[] {
  return msgs
    .filter(m => m.role === "user" || m.role === "assistant")
    .map(m => {
      let content = m.content
      if (m.role === "user" && m.hiddenContent) {
        content = m.hiddenContent + "\n\n" + content
      }
      return { role: m.role, content }
    })
}

export function useChatStream(): UseChatStreamReturn {
  const [messages, setMessages] = useState<Message[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [totalUsage, setTotalUsage] = useState<Usage>({ prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 })
  const abortRef = useRef<(() => void) | null>(null)
  const messagesRef = useRef(messages)
  messagesRef.current = messages

  const streamAssistantReply = useCallback(
    async (req: ChatRequest, history: { role: string; content: string }[], assistantMsg: Message) => {
      setIsStreaming(true)

      const batch = createFlushBatcher(setMessages, assistantMsg)
      // Every exit from the loop below settles the tool cards still marked running.
      let unresolvedReason = "Stream ended before the tool returned"

      try {
        const stream = createChatStream({ ...req, messages: history })
        abortRef.current = stream.abort

        for await (const event of stream) {
          const decoded = decodeChatStreamEvent(event)
          if (!decoded) continue
          if (decoded.kind === "usage") {
            setTotalUsage((prev) => addUsage(prev, decoded.usage))
            continue
          }
          if (decoded.kind === "done") break
          applyChatStreamEvent(assistantMsg, decoded)
          batch.schedule()
          // An error event is the turn's last content: append it, then stop reading.
          if (decoded.kind === "error") {
            unresolvedReason = decoded.message
            break
          }
        }
      } catch (e) {
        // Abort and transport failure are both swallowed: the turn keeps what it streamed.
        const err = e as Error
        unresolvedReason = err.name === "AbortError" ? "Cancelled" : err.message || "Stream failed"
      } finally {
        settleRunningToolCalls(assistantMsg, unresolvedReason)
        batch.final()
        setMessages((prev) => deactivateSentImages(prev, req.img_paths))
        setIsStreaming(false)
        abortRef.current = null
      }
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

      const history = buildHistory(messagesRef.current.slice(0, -1))
      await streamAssistantReply(req, history, assistantMsg)
    },
    [streamAssistantReply]
  )

  const regenerateAt = useCallback(
    async (userIndex: number, req: ChatRequest) => {
      const current = messagesRef.current
      const userMsg = current[userIndex]
      if (!userMsg || userMsg.role !== "user") return

      const assistantMsg: Message = { role: "assistant", content: "" }
      const truncated = current.slice(0, userIndex + 1)
      const next = [...truncated, assistantMsg]
      messagesRef.current = next
      setMessages(next)

      const history = buildHistory(truncated)
      await streamAssistantReply(req, history, assistantMsg)
    },
    [streamAssistantReply]
  )

  const cancel = useCallback(() => {
    abortRef.current?.()
    setIsStreaming(false)
  }, [])

  const deleteMessagePair = useCallback((index: number) => {
    setMessages((prev) => {
      const copy = [...prev]
      copy.splice(index, 2)
      return copy
    })
  }, [])

  const addMessagePair = useCallback((userContent: string, assistantContent: string) => {
    setMessages((prev) => [
      ...prev,
      { role: "user", content: userContent },
      { role: "assistant", content: assistantContent },
    ])
  }, [])

  return { messages, isStreaming, send, regenerateAt, cancel, deleteMessagePair, addMessagePair, setMessages, totalUsage, setTotalUsage }
}