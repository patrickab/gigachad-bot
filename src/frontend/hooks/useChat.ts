"use client"

import { useCallback, useEffect, useState } from "react"
import type React from "react"
import { fetchModels, fetchPrompts } from "@/lib/api"
import type { ChatRequest, Message, ModelsResponse, MorphicSearchParams } from "@/lib/types"
import { useChatStream } from "./useChatStream"
import { useResearch, type ResearchParams } from "./useResearch"
import { useMorphicSearch } from "./useMorphicSearch"
import { useHistory } from "./useHistory"

export type { ResearchParams }

export interface UseChatReturn {
  messages: Message[]
  isStreaming: boolean
  send: (req: ChatRequest, skipAddMessages?: boolean) => Promise<void>
  cancel: () => void
  reset: () => Promise<void>
  research: (params: ResearchParams) => Promise<void>
  morphicSearch: (params: MorphicSearchParams) => Promise<void>
  models: ModelsResponse | null
  prompts: string[]
  rootFiles: string[]
  histories: Record<string, string[]>
  historiesLoading: boolean
  refreshHistories: () => Promise<void>
  loadHistory: (filename: string) => Promise<{ messages: Message[]; chat_id: string | null }>
  deleteMessagePair: (index: number) => void
  addMessagePair: (userContent: string, assistantContent: string) => void
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>
  error: string | null
}

export function useChat(): UseChatReturn {
  const { messages, isStreaming, send, cancel, deleteMessagePair, addMessagePair, setMessages } = useChatStream()
  const { research: doResearch, error: researchError } = useResearch()
  const { morphicSearch: doMorphicSearch, error: morphicError } = useMorphicSearch()
  const { rootFiles, histories, historiesLoading, refreshHistories, loadHistory: loadHistoryRaw } = useHistory()

  const [models, setModels] = useState<ModelsResponse | null>(null)
  const [prompts, setPrompts] = useState<string[]>([])

  const error = researchError || morphicError

  useEffect(() => {
    fetchModels().then(setModels).catch(console.error)
    fetchPrompts().then(setPrompts).catch(console.error)
    refreshHistories()
  }, [])

  const appendMessage = useCallback((msg: Message) => {
    setMessages((prev) => [...prev, msg])
  }, [setMessages])

  const updateLast = useCallback((msg: Message) => {
    setMessages((prev) => {
      const copy = [...prev]
      copy[copy.length - 1] = { ...msg }
      return copy
    })
  }, [setMessages])

  const research = useCallback(async (params: ResearchParams) => {
    await doResearch(params, appendMessage, updateLast)
  }, [doResearch, appendMessage, updateLast])

  const morphicSearch = useCallback(async (params: MorphicSearchParams) => {
    await doMorphicSearch(params, appendMessage, updateLast)
  }, [doMorphicSearch, appendMessage, updateLast])

  const reset = useCallback(async () => {
    setMessages([])
  }, [setMessages])

  const loadHistory = useCallback(async (filename: string) => {
    const result = await loadHistoryRaw(filename)
    if (result.messages.length > 0) setMessages(result.messages)
    return result
  }, [loadHistoryRaw, setMessages])

  return {
    messages,
    isStreaming,
    send,
    cancel,
    reset,
    research,
    morphicSearch,
    models,
    prompts,
    rootFiles,
    histories,
    historiesLoading,
    refreshHistories,
    loadHistory,
    deleteMessagePair,
    addMessagePair,
    setMessages,
    error,
  }
}