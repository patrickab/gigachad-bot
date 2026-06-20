"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import type React from "react"
import { fetchModels, fetchPrompts, loadChatHistory as apiLoadChatHistory } from "@/lib/api"
import type { ChatRequest, Message, ModelsResponse, MorphicSearchParams, Usage } from "@/lib/types"
import { useChatStream } from "./useChatStream"
import { useResearch, type ResearchParams } from "./useResearch"
import { morphicFetch, parseMorphicStream } from "@/lib/morphic"

const MORPHIC_FLUSH_MS = 60

export type { ResearchParams }

export interface UseChatReturn {
  messages: Message[]
  isStreaming: boolean
  send: (req: ChatRequest, skipAddMessages?: boolean) => Promise<void>
  regenerateAt: (userIndex: number, req: ChatRequest) => Promise<void>
  cancel: () => void
  reset: () => Promise<void>
  research: (params: ResearchParams) => Promise<void>
  morphicSearch: (params: MorphicSearchParams) => Promise<void>
  models: ModelsResponse | null
  prompts: Record<string, string>
  loadHistory: (filename: string) => Promise<{ messages: Message[]; chat_id: string | null; parent_id: string | null; branch_message_idx: number | null }>
  deleteMessagePair: (index: number) => void
  addMessagePair: (userContent: string, assistantContent: string) => void
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>
  error: string | null
  totalUsage: Usage
  setTotalUsage: React.Dispatch<React.SetStateAction<Usage>>
}

export function useChat(): UseChatReturn {
  const { messages, isStreaming, send, regenerateAt, cancel: cancelStream, deleteMessagePair, addMessagePair, setMessages, totalUsage, setTotalUsage } = useChatStream()
  const { research: doResearch, error: researchError } = useResearch()

  const [models, setModels] = useState<ModelsResponse | null>(null)
  const [prompts, setPrompts] = useState<Record<string, string>>({})
  const [morphicError, setMorphicError] = useState<string | null>(null)
  const [morphicSearching, setMorphicSearching] = useState(false)
  const morphicAbortRef = useRef<(() => void) | null>(null)

  const error = researchError || morphicError

  useEffect(() => {
    fetchModels().then(setModels).catch(console.error)
    fetchPrompts().then(setPrompts).catch(console.error)
  }, [])

  const appendMessage = useCallback((msg: Message) => {
    setMessages((prev) => [...prev, msg])
  }, [setMessages])

  const updateLast = useCallback((msg: Message) => {
    setMessages((prev) => {
      const last = prev[prev.length - 1]
      if (!last || last.role !== "assistant") return prev
      const copy = [...prev]
      copy[copy.length - 1] = { ...msg }
      return copy
    })
  }, [setMessages])

  const research = useCallback(async (params: ResearchParams) => {
    await doResearch(params, appendMessage, updateLast)
  }, [doResearch, appendMessage, updateLast])

  const morphicSearch = useCallback(async (params: MorphicSearchParams) => {
    setMorphicError(null)
    setMorphicSearching(true)

    const userMsg: Message = { role: "user", content: params.query }
    const assistantMsg: Message = { role: "assistant", content: "" }

    appendMessage(userMsg)
    appendMessage(assistantMsg)

    let lastFlush = 0
    let pending = false
    let flushTimer: ReturnType<typeof setTimeout> | null = null

    const flush = () => {
      pending = false
      lastFlush = performance.now()
      setMessages((prev) => {
        const last = prev[prev.length - 1]
        if (!last || last.role !== "assistant") return prev
        const copy = [...prev]
        copy[copy.length - 1] = {
          role: "assistant",
          content: assistantMsg.content,
          morphic_result: assistantMsg.morphic_result
            ? {
                query: assistantMsg.morphic_result.query,
                sources: assistantMsg.morphic_result.sources,
                images: assistantMsg.morphic_result.images,
                citationMap: assistantMsg.morphic_result.citationMap,
              }
            : undefined,
        }
        return copy
      })
    }

    const scheduleFlush = () => {
      if (pending) return
      const now = performance.now()
      const delay = Math.max(0, MORPHIC_FLUSH_MS - (now - lastFlush))
      if (delay <= 0) {
        flush()
      } else {
        pending = true
        flushTimer = setTimeout(flush, delay)
      }
    }

    try {
      const { promise, abort } = morphicFetch(params)
      morphicAbortRef.current = abort

      const res = await promise

      for await (const event of parseMorphicStream(res)) {
        if (event.type === "text" && event.text) {
          assistantMsg.content += event.text
          scheduleFlush()
        } else if (event.type === "source") {
          const prev = assistantMsg.morphic_result
          assistantMsg.morphic_result = {
            query: event.query ?? prev?.query ?? params.query,
            sources: event.sources ? [...(prev?.sources ?? []), ...event.sources] : (prev?.sources ?? []),
            images: event.images ? [...new Set([...(prev?.images ?? []), ...event.images])] : (prev?.images ?? []),
            citationMap: event.citationMap ?? prev?.citationMap,
          }
          scheduleFlush()
        } else if (event.type === "error") {
          throw new Error(event.text || "Morphic search error")
        }
      }
    } catch (e) {
      if ((e as Error).name === "AbortError") return
      const msg = (e as Error)?.message ?? "Search failed"
      setMorphicError(msg)
      assistantMsg.content = assistantMsg.content || `Search error: ${msg}`
      if (flushTimer) clearTimeout(flushTimer)
      flush()
    } finally {
      if (flushTimer) clearTimeout(flushTimer)
      flush()
      morphicAbortRef.current = null
      setMorphicSearching(false)
    }
  }, [appendMessage, setMessages])

  const cancel = useCallback(() => {
    cancelStream()
    morphicAbortRef.current?.()
  }, [cancelStream])

  const reset = useCallback(async () => {
    morphicAbortRef.current?.()
    setMessages([])
    setTotalUsage({ prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 })
  }, [setMessages, setTotalUsage])

  const loadHistory = useCallback(async (filename: string) => {
    try {
      const data = await apiLoadChatHistory(filename)
      return { messages: data.messages ?? [], chat_id: data.chat_id ?? null, parent_id: data.parent_id ?? null, branch_message_idx: data.branch_message_idx ?? null }
    } catch {
      return { messages: [] as Message[], chat_id: null as string | null, parent_id: null as string | null, branch_message_idx: null as number | null }
    }
  }, [])

  return {
    messages,
    isStreaming: isStreaming || morphicSearching,
    send,
    regenerateAt,
    cancel,
    reset,
    research,
    morphicSearch,
    models,
    prompts,
    loadHistory,
    deleteMessagePair,
    addMessagePair,
    setMessages,
    error,
    totalUsage,
    setTotalUsage,
  }
}
