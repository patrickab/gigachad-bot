/** LiteLLM prefix used for OMP-sourced models: `litellm_proxy/<omp-provider>/<id>`. */
export const OMP_LITELLM_ID = "litellm_proxy"

export function displayName(m: string): string {
  // OMP models all live under one provider tab, so keep the OMP provider
  // segment on screen and drop only the LiteLLM proxy hop: a row reads
  // `anthropic/claude-opus-5`, not a bare model id shared across providers.
  if (m.startsWith(`${OMP_LITELLM_ID}/`)) return m.slice(OMP_LITELLM_ID.length + 1)
  const slash = m.indexOf("/")
  return slash === -1 ? m : m.slice(slash + 1)
}
