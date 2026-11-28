"use client"

import { useResource, useResourceState } from "@/hooks/useResource"
import { saveModelDefaults, saveModelProviders, saveModelTabOrder } from "@/lib/backend/api"
import { modelsResource, promptsResource } from "@/lib/backend/resources"
import type { ModelsResponse } from "@/lib/domain/types"

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

/** The model and prompt config every tab shares. */
export function useChatConfig() {
  const { value: models = null, error: modelLoadError } = useResourceState("Models", modelsResource)
  const prompts = useResource("Prompts", promptsResource) ?? NO_PROMPTS
  return { ...modelSaves, models, modelLoadError, prompts }
}
