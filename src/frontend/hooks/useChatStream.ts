"use client"

import { useCallback, useRef, useState } from "react"
import { createChatStream } from "@/lib/api"
import type { ChatRequest, Message } from "@/lib/types"

export interface UseChatStreamReturn {
  messages: Message[]
  isStreaming: boolean
  send: (req: ChatRequest) => Promise<void>
  cancel: () => void
  deleteMessagePair: (index: number) => void
  addMessagePair: (userContent: string, assistantContent: string) => void
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>
}

export function useChatStream(): UseChatStreamReturn {
  const [messages, setMessages] = useState<Message[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const abortRef = useRef<(() => void) | null>(null)
  const assistantRef = useRef<Message | null>(null)

  const send = useCallback(
    async (req: ChatRequest) => {
      setIsStreaming(true)

      const userMsg: Message = { role: "user", content: req.user_msg }
      const assistantMsg: Message = { role: "assistant", content: "" }
      assistantRef.current = assistantMsg

      setMessages((prev) => [...prev, userMsg, assistantMsg])

      try {
        const { stream, abort } = createChatStream({ ...req, messages: [] })
        abortRef.current = abort

        const reader = stream
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          const text = new TextDecoder().decode(value, { stream: true })
          if (text) {
            assistantMsg.content += text
            setMessages((prev) => {
              const copy = [...prev]
              copy[copy.length - 1] = { ...assistantMsg }
              return copy
            })
          }
        }
      } catch (e) {
        if ((e as Error).name === "AbortError") return
      } finally {
        setIsStreaming(false)
        abortRef.current = null
        assistantRef.current = null
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

  return { messages, isStreaming, send, cancel, deleteMessagePair, addMessagePair, setMessages }
}