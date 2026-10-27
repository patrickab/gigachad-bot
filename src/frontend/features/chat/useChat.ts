"use client"

import { useCallback, useState } from "react"
import { useStartupRead } from "@/contexts/StartupReadContext"
import { fetchModels, fetchPrompts, saveModelDefaults as saveModelDefaultsRequest, saveModelProviders as saveModelProvidersRequest, saveModelTabOrder as saveModelTabOrderRequest } from "@/lib/backend/api"
import type { ModelDefaults, ModelProvider, ModelsResponse } from "@/lib/domain/types"
import { useChatStream } from "./useChatStream"

/** Compose chat streaming with the model and prompt config loaded on mount. */
export function useChat() {
  const stream = useChatStream()
  const { setMessages, setTotalUsage } = stream

  const [models, setModels] = useState<ModelsResponse | null>(null)
  const [prompts, setPrompts] = useState<Record<string, string>>({})

  const { error: modelLoadError } = useStartupRead("Models", fetchModels, setModels)
  useStartupRead("Prompts", fetchPrompts, setPrompts)

  const saveModelProviders = useCallback(async (providers: ModelProvider[]) => setModels(await saveModelProvidersRequest(providers)), [])
  const saveModelTabOrder = useCallback(async (order: string[]) => setModels(await saveModelTabOrderRequest(order)), [])
  const saveModelDefaults = useCallback(async (defaults: ModelDefaults) => setModels(await saveModelDefaultsRequest(defaults)), [])

  const reset = useCallback(() => {
    setMessages([])
    setTotalUsage({ prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 })
  }, [setMessages, setTotalUsage])

  return { ...stream, reset, models, modelLoadError, saveModelProviders, saveModelTabOrder, saveModelDefaults, prompts, setPrompts }
}
