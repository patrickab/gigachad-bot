import type { ModelProvider } from "@/lib/domain/types"

/** LiteLLM prefix used for Pi-sourced models: `litellm_proxy/<pi-provider>/<id>`. */
export const PI_LITELLM_ID = "litellm_proxy"
/** `source` of the catalog row holding every Pi gateway pick (backend `pi_models.SOURCE`). */
export const PI_MODEL_SOURCE = "pi"
/** Label of that row, one selector tab listing `<pi-provider>/<model>` (backend `pi_models.PROVIDER_LABEL`). */
export const PI_PROVIDER_LABEL = "Pi"

/** A provider's models as `litellm_id/model` ids. */
export function qualifiedModelIds(provider: ModelProvider): string[] {
  return provider.models.map((model) => `${provider.litellm_id}/${model}`)
}

export function displayName(m: string): string {
  // Keep the Pi provider segment (`anthropic/claude-opus-5`) so rows stay distinct under the
  // single Pi provider tab. Drop only the LiteLLM proxy hop.
  if (m.startsWith(`${PI_LITELLM_ID}/`)) return m.slice(PI_LITELLM_ID.length + 1)
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
