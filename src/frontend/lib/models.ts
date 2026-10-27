import type { ModelProvider } from "@/lib/domain/types"

/** LiteLLM prefix used for OMP-sourced models: `litellm_proxy/<omp-provider>/<id>`. */
export const OMP_LITELLM_ID = "litellm_proxy"

/** A provider's models as `litellm_id/model` ids. */
export function qualifiedModelIds(provider: ModelProvider): string[] {
  return provider.models.map((model) => `${provider.litellm_id}/${model}`)
}

export function displayName(m: string): string {
  // Keep the OMP provider segment (`anthropic/claude-opus-5`) so rows stay distinct under the
  // single OMP provider tab. Drop only the LiteLLM proxy hop.
  if (m.startsWith(`${OMP_LITELLM_ID}/`)) return m.slice(OMP_LITELLM_ID.length + 1)
  const slash = m.indexOf("/")
  return slash === -1 ? m : m.slice(slash + 1)
}

/** `providers` in the saved tab order. Unlisted providers (never dragged, or new
 *  since the order was saved) fall to the end, alphabetical, the same rank rule
 *  PromptStore uses for prompt display order. */
export function orderProviders<T extends { label: string }>(providers: T[], tabOrder: string[]): T[] {
  return [...providers].sort((a, b) => {
    const ai = tabOrder.indexOf(a.label)
    const bi = tabOrder.indexOf(b.label)
    if (ai !== -1 && bi !== -1) return ai - bi
    if (ai !== -1) return -1
    if (bi !== -1) return 1
    return a.label.localeCompare(b.label)
  })
}
