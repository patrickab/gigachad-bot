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
