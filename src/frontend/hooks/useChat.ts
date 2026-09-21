"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import type React from "react"
import { fetchModels, fetchPrompts, saveModelDefaults as saveModelDefaultsRequest, saveModelProviders as saveModelProvidersRequest, saveModelTabOrder as saveModelTabOrderRequest } from "@/lib/api"
import type { ChatRequest, Message, ModelDefaults, ModelProvider, ModelsResponse, WebSearchParams, Usage } from "@/lib/types"
import { useChatStream, type ChatStreamCompletion } from "./useChatStream"
import { useResearch, type ResearchParams } from "./useResearch"
import { webSearchFetch, parseWebSearchStream, type WebSearchResultItem } from "@/lib/webSearch"
import { createFlushBatcher } from "@/lib/streaming"

export type { ResearchParams }

export interface UseChatReturn {
  messages: Message[]
  isStreaming: boolean
  send: (req: ChatRequest, skipAddMessages?: boolean) => Promise<ChatStreamCompletion | null>
  regenerateAt: (userIndex: number, req: ChatRequest) => Promise<ChatStreamCompletion | null>
  cancel: () => void
  reset: () => Promise<void>
  research: (params: ResearchParams) => Promise<void>
  webSearch: (params: WebSearchParams) => Promise<void>
  models: ModelsResponse | null
  saveModelProviders: (providers: ModelProvider[]) => Promise<void>
  saveModelTabOrder: (order: string[]) => Promise<void>
  saveModelDefaults: (defaults: ModelDefaults) => Promise<void>
  prompts: Record<string, string>
  setPrompts: React.Dispatch<React.SetStateAction<Record<string, string>>>
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

  const saveModelProviders = useCallback(async (providers: ModelProvider[]) => {
    setModels(await saveModelProvidersRequest(providers))
  }, [])
  const saveModelTabOrder = useCallback(async (order: string[]) => {
    setModels(await saveModelTabOrderRequest(order))
  }, [])
  const saveModelDefaults = useCallback(async (defaults: ModelDefaults) => {
    setModels(await saveModelDefaultsRequest(defaults))
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

    // Source labels arrive before text and remain stable for the saved citation map.
    const setSources = (sources: WebSearchResultItem[]) => {
      assistantMsg.search_result = {
        query: params.query,
        sources,
        images: assistantMsg.search_result?.images ?? [],
        videos: assistantMsg.search_result?.videos ?? [],
        citationMap: Object.fromEntries(sources.map((s, i) => [s.label ?? String(i + 1), s])),
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
    saveModelProviders,
    saveModelTabOrder,
    saveModelDefaults,
    prompts,
    setPrompts,
    deleteMessagePair,
    addMessagePair,
    setMessages,
    error,
    totalUsage,
    setTotalUsage,
  }
}
