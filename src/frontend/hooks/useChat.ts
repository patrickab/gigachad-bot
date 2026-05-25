"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { createChatStream, fetchHistory, fetchModels, fetchPrompts, listChatHistories, loadChatHistory as loadApiHistory, resetHistory } from "@/lib/api"
import type { ChatRequest, Message, ModelsResponse } from "@/lib/types"

export interface UseChatReturn {
  messages: Message[]
  isStreaming: boolean
  send: (req: ChatRequest) => Promise<void>
  cancel: () => void
  reset: () => Promise<void>
  models: ModelsResponse | null
  prompts: string[]
  histories: Record<string, string[]>
  refreshHistories: () => Promise<void>
  error: string | null
  loadHistory: (filename: string) => Promise<void>
  deleteMessagePair: (index: number) => void
}

export function useChat(): UseChatReturn {
  const [messages, setMessages] = useState<Message[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [models, setModels] = useState<ModelsResponse | null>(null)
  const [prompts, setPrompts] = useState<string[]>([])
  const [histories, setHistories] = useState<Record<string, string[]>>({})
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<(() => void) | null>(null)
  const assistantRef = useRef<Message | null>(null)

  useEffect(() => {
    fetchModels().then(setModels).catch(console.error)
    fetchPrompts().then(setPrompts).catch(console.error)
    loadInitialHistory()
    loadHistories()
  }, [])

  async function loadInitialHistory() {
    try {
      const data = await fetchHistory()
      setMessages(data.messages ?? [])
    } catch {
      // no saved state
    }
  }

  async function loadHistories() {
    try {
      const data = await listChatHistories()
      setHistories(data.histories ?? {})
    } catch {
      // unavailable
    }
  }

  const send = useCallback(
    async (req: ChatRequest) => {
      setError(null)
      setIsStreaming(true)

      const userMsg: Message = { role: "user", content: req.user_msg }
      const assistantMsg: Message = { role: "assistant", content: "" }
      assistantRef.current = assistantMsg

      setMessages((prev) => [...prev, userMsg, assistantMsg])

      try {
        const { stream, abort } = createChatStream(req)
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
        setError((e as Error).message || "Stream error")
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

  const reset = useCallback(async () => {
    await resetHistory()
    setMessages([])
  }, [])

  const loadHistoryMessages = useCallback(async (filename: string) => {
    try {
      const data = await loadApiHistory(filename)
      setMessages(data.messages ?? [])
    } catch (e) {
      setError((e as Error).message)
    }
  }, [])

  const deleteMessagePair = useCallback((index: number) => {
    setMessages((prev) => {
      const copy = [...prev]
      copy.splice(index, 2)
      return copy
    })
  }, [])

  return {
    messages,
    isStreaming,
    send,
    cancel,
    reset,
    models,
    prompts,
    histories,
    refreshHistories: loadHistories,
    error,
    loadHistory: loadHistoryMessages,
    deleteMessagePair,
  }
}
