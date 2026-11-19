import type { ModelProvider } from "@/lib/domain/types"

/** Stored selector prefix of Pi gateway models, kept from before Pi: `litellm_proxy/<pi-provider>/<id>`. */
export const PI_LITELLM_ID = "litellm_proxy"
/** `source` of the catalog row holding every Pi gateway pick (backend `pi_models.SOURCE`). */
export const PI_MODEL_SOURCE = "pi"
/** Label of that row, one selector tab listing `<pi-provider>/<model>` (backend `pi_models.PROVIDER_LABEL`). */
export const PI_PROVIDER_LABEL = "Pi"
/** Stored selectors keep their pre-Pi provider ids, the gateway names these providers differently. */
const GATEWAY_PROVIDER_ALIASES: Record<string, string> = { gemini: "google" }

/** The gateway's `<provider>/<id>` of a stored selector, mirror of backend `pi_models.gateway_model_id`: `litellm_proxy/x` -> `x`, `gemini/m` -> `google/m`. */
export function gatewayModelId(selector: string): string {
  if (selector.startsWith(`${PI_LITELLM_ID}/`)) return selector.slice(PI_LITELLM_ID.length + 1)
  const slash = selector.indexOf("/")
  if (slash === -1) return selector
  const provider = selector.slice(0, slash)
  return `${Object.hasOwn(GATEWAY_PROVIDER_ALIASES, provider) ? GATEWAY_PROVIDER_ALIASES[provider] : provider}${selector.slice(slash)}`
}

/** A provider's models as `provider_id/model` ids. */
export function qualifiedModelIds(provider: ModelProvider): string[] {
  return provider.models.map((model) => `${provider.provider_id}/${model}`)
}

export function displayName(m: string): string {
  // Keep the Pi provider segment (`anthropic/claude-opus-5`) so rows stay distinct under the
  // single Pi provider tab. Drop only the stored `litellm_proxy/` prefix.
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
