import type { LearnView, ModelsResponse } from "@/lib/domain/types"
import type { LearnFrameMessageOf, LearnModelRef } from "@gigachad/chat-ui/learn-protocol"

export type LearnModelState = LearnFrameMessageOf<"model">

/** Whether frame state for `sessionId` belongs to `view`. Home and `/chat/new` both report null (the draft's strong tier). */
export function isFrameStateOf(view: LearnView, sessionId: string | null): boolean {
  return sessionId === (view.kind === "session" ? view.sessionId : null)
}

/**
 * Selector models for learn tabs: only what LearnHarness's proxy serves (`models.served`). A served model's id is the
 * gigachad selector string, so a catalog entry the proxy does not serve is left out (it arrives after a proxy restart).
 */
export function learnSelectorModels(models: ModelsResponse, served: readonly LearnModelRef[]): ModelsResponse {
  const ids = new Set(served.map((model) => model.modelId))
  return {
    ...models,
    ollama: models.ollama.filter((id) => ids.has(id)),
    providers: models.providers.map((provider) => ({ ...provider, models: provider.models.filter((model) => ids.has(`${provider.litellm_id}/${model}`)) })),
  }
}

/** The served model a selector id names, null for one the proxy does not serve. */
export function servedModelOf(served: readonly LearnModelRef[], selectorId: string): LearnModelRef | null {
  return served.find((model) => model.modelId === selectorId) ?? null
}
