"use client"

import { useCallback, useMemo, useState } from "react"
import { useStartupRead } from "@/contexts/StartupReadContext"
import { fetchModels, fetchPrompts, saveModelDefaults, saveModelProviders, saveModelTabOrder } from "@/lib/backend/api"
import type { ModelsResponse } from "@/lib/domain/types"
import { useChatStream } from "./useChatStream"

/** `save` whose response becomes the loaded models. */
function applyModels<A extends unknown[]>(setModels: (models: ModelsResponse) => void, save: (...args: A) => Promise<ModelsResponse>) {
  return async (...args: A) => setModels(await save(...args))
}

/** Compose chat streaming with the model and prompt config loaded on mount. */
export function useChat() {
  const stream = useChatStream()
  const { setMessages, setTotalUsage } = stream

  const [models, setModels] = useState<ModelsResponse | null>(null)
  const [prompts, setPrompts] = useState<Record<string, string>>({})

  const { error: modelLoadError } = useStartupRead("Models", fetchModels, setModels)
  useStartupRead("Prompts", fetchPrompts, setPrompts)

  const modelSaves = useMemo(() => ({
    saveModelProviders: applyModels(setModels, saveModelProviders),
    saveModelTabOrder: applyModels(setModels, saveModelTabOrder),
    saveModelDefaults: applyModels(setModels, saveModelDefaults),
  }), [])

  const reset = useCallback(() => {
    setMessages([])
    setTotalUsage({ prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 })
  }, [setMessages, setTotalUsage])

  return { ...stream, ...modelSaves, reset, models, modelLoadError, prompts, setPrompts }
}
