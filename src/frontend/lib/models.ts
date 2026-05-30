import type { Provider } from "./types"

export const PROVIDER_PREFIXES: Record<Provider, string> = {
  Ollama: "ollama/",
  Gemini: "gemini/",
  OpenAI: "openai/",
}

export function displayName(m: string): string {
  for (const prefix of Object.values(PROVIDER_PREFIXES)) {
    if (m.startsWith(prefix)) return m.slice(prefix.length)
  }
  return m
}

export function groupByProvider(
  models: { ollama: string[]; gemini: string[]; openai: string[] } | null
): Record<Provider, string[]> {
  if (!models) return { Ollama: [], Gemini: [], OpenAI: [] }
  return {
    Ollama: models.ollama ?? [],
    Gemini: models.gemini ?? [],
    OpenAI: models.openai ?? [],
  }
}

export function activeProviders(grouped: Record<Provider, string[]>): Provider[] {
  return (Object.keys(grouped) as Provider[]).filter((k) => grouped[k].length > 0)
}