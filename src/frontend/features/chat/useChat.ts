"use client"

import { useCallback } from "react"
import { useResource, useResourceState } from "@/hooks/useResource"
import { saveModelDefaults, saveModelProviders, saveModelTabOrder } from "@/lib/backend/api"
import { modelsResource, promptsResource } from "@/lib/backend/resources"
import type { ModelsResponse } from "@/lib/domain/types"
import { useChatStream } from "./useChatStream"

const NO_PROMPTS: Record<string, string> = {}

/** `save` whose response becomes every tab's models. */
function sharingModels<A extends unknown[]>(save: (...args: A) => Promise<ModelsResponse>) {
  return async (...args: A) => modelsResource.set(await save(...args))
}

const modelSaves = {
  saveModelProviders: sharingModels(saveModelProviders),
  saveModelTabOrder: sharingModels(saveModelTabOrder),
  saveModelDefaults: sharingModels(saveModelDefaults),
}

/** Compose chat streaming with the model and prompt config every tab shares. */
export function useChat() {
  const stream = useChatStream()
  const { setMessages, setTotalUsage } = stream

  const { value: models = null, error: modelLoadError } = useResourceState("Models", modelsResource)
  const prompts = useResource("Prompts", promptsResource) ?? NO_PROMPTS

  const reset = useCallback(() => {
    setMessages([])
    setTotalUsage({ prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 })
  }, [setMessages, setTotalUsage])

  return { ...stream, ...modelSaves, reset, models, modelLoadError, prompts }
}
