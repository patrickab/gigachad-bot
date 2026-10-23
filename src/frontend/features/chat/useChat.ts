"use client"

import { useCallback, useState } from "react"
import type React from "react"
import { useStartupRead } from "@/contexts/StartupReadContext"
import { fetchModels, fetchPrompts, saveModelDefaults as saveModelDefaultsRequest, saveModelProviders as saveModelProvidersRequest, saveModelTabOrder as saveModelTabOrderRequest } from "@/lib/backend/api"
import type { ChatRequest, Message, ModelDefaults, ModelProvider, ModelsResponse, Usage } from "@/lib/domain/types"
import { useChatStream, type ChatStreamCompletion, type UseChatStreamReturn } from "./useChatStream"

interface UseChatReturn {
  messages: Message[]
  isStreaming: boolean
  send: (req: ChatRequest, skipAddMessages?: boolean) => Promise<ChatStreamCompletion | null>
  regenerateAt: (userIndex: number, req: ChatRequest) => Promise<ChatStreamCompletion | null>
  cancel: () => void
  reset: () => Promise<void>
  models: ModelsResponse | null
  modelLoadError: string | null
  saveModelProviders: (providers: ModelProvider[]) => Promise<void>
  saveModelTabOrder: (order: string[]) => Promise<void>
  saveModelDefaults: (defaults: ModelDefaults) => Promise<void>
  prompts: Record<string, string>
  setPrompts: React.Dispatch<React.SetStateAction<Record<string, string>>>
  deleteMessagePair: (index: number) => void
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>
  replaceMessages: UseChatStreamReturn["replaceMessages"]
  totalUsage: Usage
  setTotalUsage: React.Dispatch<React.SetStateAction<Usage>>
}

/** Compose chat streaming with the model and prompt config loaded on mount. */
export function useChat(): UseChatReturn {
  const { messages, isStreaming, send, regenerateAt, cancel, deleteMessagePair, setMessages, replaceMessages, totalUsage, setTotalUsage } = useChatStream()

  const [models, setModels] = useState<ModelsResponse | null>(null)
  const [prompts, setPrompts] = useState<Record<string, string>>({})

  const { error: modelLoadError } = useStartupRead("Models", fetchModels, setModels)
  useStartupRead("Prompts", fetchPrompts, setPrompts)

  const saveModelProviders = useCallback(async (providers: ModelProvider[]) => {
    setModels(await saveModelProvidersRequest(providers))
  }, [])
  const saveModelTabOrder = useCallback(async (order: string[]) => {
    setModels(await saveModelTabOrderRequest(order))
  }, [])
  const saveModelDefaults = useCallback(async (defaults: ModelDefaults) => {
    setModels(await saveModelDefaultsRequest(defaults))
  }, [])

  const reset = useCallback(async () => {
    setMessages([])
    setTotalUsage({ prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 })
  }, [setMessages, setTotalUsage])

  return {
    messages,
    isStreaming,
    send,
    regenerateAt,
    cancel,
    reset,
    models,
    modelLoadError,
    saveModelProviders,
    saveModelTabOrder,
    saveModelDefaults,
    prompts,
    setPrompts,
    deleteMessagePair,
    setMessages,
    replaceMessages,
    totalUsage,
    setTotalUsage,
  }
}
