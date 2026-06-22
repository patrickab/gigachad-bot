"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import type React from "react"
import { fetchModels, fetchPrompts, loadChatHistory as apiLoadChatHistory } from "@/lib/api"
import type { ChatRequest, Message, ModelsResponse, WebSearchParams, Usage } from "@/lib/types"
import { useChatStream } from "./useChatStream"
import { useResearch, type ResearchParams } from "./useResearch"
import { webSearchFetch, parseWebSearchStream, fetchWebSearchImages, fetchWebSearchVideos, type WebSearchResultItem } from "@/lib/webSearch"
import { createFlushBatcher } from "@/lib/streaming"

export type { ResearchParams }

export interface UseChatReturn {
  messages: Message[]
  isStreaming: boolean
  send: (req: ChatRequest, skipAddMessages?: boolean) => Promise<void>
  regenerateAt: (userIndex: number, req: ChatRequest) => Promise<void>
  cancel: () => void
  reset: () => Promise<void>
  research: (params: ResearchParams) => Promise<void>
  webSearch: (params: WebSearchParams) => Promise<void>
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
  const [searchError, setSearchError] = useState<string | null>(null)
  const [searching, setSearching] = useState(false)
  const searchAbortRef = useRef<(() => void) | null>(null)

  const error = researchError || searchError

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

  const webSearch = useCallback(async (params: WebSearchParams) => {
    setSearchError(null)
    setSearching(true)

    const userMsg: Message = { role: "user", content: params.query }
    const assistantMsg: Message = { role: "assistant", content: "" }

    appendMessage(userMsg)
    appendMessage(assistantMsg)

    const batch = createFlushBatcher(setMessages, assistantMsg)

    // Sources arrive as a list; build a 1-based citation map so [n] markers resolve.
    const setSources = (sources: WebSearchResultItem[]) => {
      assistantMsg.search_result = {
        query: params.query,
        sources,
        images: assistantMsg.search_result?.images ?? [],
        videos: assistantMsg.search_result?.videos ?? [],
        citationMap: Object.fromEntries(sources.map((s, i) => [String(i + 1), s])),
      }
    }

    try {
      const { promise, abort } = webSearchFetch(params)
      searchAbortRef.current = abort

      const res = await promise

      for await (const event of parseWebSearchStream(res)) {
        if (event.type === "text" && event.text) {
          assistantMsg.content += event.text
          batch.schedule()
        } else if (event.type === "sources") {
          const prev = assistantMsg.search_result?.sources ?? []
          setSources([...prev, ...(event.sources ?? [])])
          batch.schedule()
        } else if (event.type === "error") {
          throw new Error(event.text || "Web search error")
        }
      }

      // Image/video search are separate Vane endpoints, fired only when toggled on.
      if (params.images || params.videos) {
        const [images, videos] = await Promise.all([
          params.images ? fetchWebSearchImages(params.query, params.model ?? "") : Promise.resolve([]),
          params.videos ? fetchWebSearchVideos(params.query, params.model ?? "") : Promise.resolve([]),
        ])
        assistantMsg.search_result = {
          query: params.query,
          sources: assistantMsg.search_result?.sources ?? [],
          citationMap: assistantMsg.search_result?.citationMap,
          images,
          videos,
        }
        batch.schedule()
      }
    } catch (e) {
      if ((e as Error).name === "AbortError") return
      const msg = (e as Error)?.message ?? "Search failed"
      setSearchError(msg)
      assistantMsg.content = assistantMsg.content || `Search error: ${msg}`
    } finally {
      batch.final()
      searchAbortRef.current = null
      setSearching(false)
    }
  }, [appendMessage, setMessages])

  const cancel = useCallback(() => {
    cancelStream()
    searchAbortRef.current?.()
  }, [cancelStream])

  const reset = useCallback(async () => {
    searchAbortRef.current?.()
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
    isStreaming: isStreaming || searching,
    send,
    regenerateAt,
    cancel,
    reset,
    research,
    webSearch,
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
