import type { LearnView, ModelsResponse } from "@/lib/domain/types"
import type { LearnFrameMessageOf, LearnModelRef } from "@gigachad/chat-ui/learn-protocol"
import { gatewayModelId, qualifiedModelIds } from "@/lib/models"

export type LearnModelState = LearnFrameMessageOf<"model">

/** Whether frame state for `sessionId` belongs to `view`. Home and `/chat/new` both report null (the draft's strong tier). */
export function isFrameStateOf(view: LearnView, sessionId: string | null): boolean {
  return sessionId === (view.kind === "session" ? view.sessionId : null)
}

/**
 * Selector models for learn tabs: only what LearnHarness serves (`models.served`). A served model's id is the Pi gateway
 * id of a gigachad selector (`gatewayModelId`), so a catalog entry the gateway does not serve is left out (it arrives
 * after a LearnHarness restart).
 */
export function learnSelectorModels(models: ModelsResponse, served: readonly LearnModelRef[]): ModelsResponse {
  const ids = new Set(served.map((model) => model.modelId))
  const isServed = (selectorId: string) => ids.has(gatewayModelId(selectorId))
  return {
    ...models,
    ollama: models.ollama.filter(isServed),
    providers: models.providers.map((provider) => ({ ...provider, models: provider.models.filter((model) => isServed(`${provider.provider_id}/${model}`)) })),
  }
}

/** The served model a selector id names, null for one LearnHarness does not serve. */
export function servedModelOf(served: readonly LearnModelRef[], selectorId: string): LearnModelRef | null {
  const modelId = gatewayModelId(selectorId)
  return served.find((model) => model.modelId === modelId) ?? null
}

/** The catalog's selector id for LearnHarness's model id, the model id itself when no catalog entry names it. */
export function selectorIdOf(models: ModelsResponse, modelId: string): string {
  const selectorIds = [...models.providers.flatMap(qualifiedModelIds), ...models.ollama]
  return selectorIds.find((selectorId) => gatewayModelId(selectorId) === modelId) ?? modelId
}
