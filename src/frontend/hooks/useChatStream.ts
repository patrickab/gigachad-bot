"use client"

import { useCallback, useRef, useState } from "react"
import { createChatStream } from "@/lib/api"
import type { ChatRequest, Message, Usage } from "@/lib/types"

export interface UseChatStreamReturn {
  messages: Message[]
  isStreaming: boolean
  send: (req: ChatRequest, skipAddMessages?: boolean) => Promise<void>
  cancel: () => void
  deleteMessagePair: (index: number) => void
  addMessagePair: (userContent: string, assistantContent: string) => void
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>
  totalUsage: Usage
  setTotalUsage: React.Dispatch<React.SetStateAction<Usage>>
}

const FLUSH_MS = 60

export function useChatStream(): UseChatStreamReturn {
  const [messages, setMessages] = useState<Message[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [totalUsage, setTotalUsage] = useState<Usage>({ prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 })
  const abortRef = useRef<(() => void) | null>(null)

  const send = useCallback(
    async (req: ChatRequest, skipAddMessages = false) => {
      setIsStreaming(true)

      const assistantMsg: Message = { role: "assistant", content: "" }

      if (!skipAddMessages) {
        setMessages((prev) => [
          ...prev,
          { role: "user", content: req.user_msg },
          assistantMsg,
        ])
      }

      let lastFlush = 0
      let pending = false

      const flush = () => {
        pending = false
        lastFlush = performance.now()
        setMessages((prev) => {
          const copy = [...prev]
          copy[copy.length - 1] = { ...assistantMsg }
          return copy
        })
      }

      const scheduleFlush = () => {
        if (pending) return
        const now = performance.now()
        const delay = Math.max(0, FLUSH_MS - (now - lastFlush))
        if (delay <= 0) {
          flush()
        } else {
          pending = true
          setTimeout(flush, delay)
        }
      }

      try {
        const stream = createChatStream({ ...req, messages: [] })
        abortRef.current = stream.abort

        for await (const event of stream) {
          if (event.event === "token") {
            assistantMsg.content += event.data
            scheduleFlush()
          } else if (event.event === "usage") {
            const turn: Usage = JSON.parse(event.data)
            setTotalUsage((prev) => ({
              prompt_tokens: prev.prompt_tokens + turn.prompt_tokens,
              completion_tokens: prev.completion_tokens + turn.completion_tokens,
              total_tokens: prev.total_tokens + turn.total_tokens,
            }))
          } else if (event.event === "done") {
            break
          } else if (event.event === "error") {
            assistantMsg.content += `\n\nError: ${event.data}`
            scheduleFlush()
            break
          }
        }
      } catch (e) {
        if ((e as Error).name === "AbortError") return
      } finally {
        setMessages((prev) => {
          const copy = [...prev]
          copy[copy.length - 1] = { ...assistantMsg }
          return copy
        })
        setIsStreaming(false)
        abortRef.current = null
      }
    },
    []
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

  return { messages, isStreaming, send, cancel, deleteMessagePair, addMessagePair, setMessages, totalUsage, setTotalUsage }
}