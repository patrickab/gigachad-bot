// Model catalog, model settings and prompt routes.
import { toQuery } from "../urls"
import { del, put, request } from "./http"
import type { ModelDefaults, ModelProvider, ModelsResponse, OmpCatalog, PromptMeta, ReasoningLevels } from "@/lib/domain/types"

export async function fetchModels(signal?: AbortSignal): Promise<ModelsResponse> {
  return request<ModelsResponse>("/models", { signal })
}

export async function fetchReasoningLevels(model: string, signal?: AbortSignal): Promise<ReasoningLevels> {
  return request<ReasoningLevels>(`/models/reasoning-levels${toQuery({ model })}`, { signal })
}

export async function saveModelProviders(providers: ModelProvider[]): Promise<ModelsResponse> {
  return put<ModelsResponse>("/models/providers", {
    providers: Object.fromEntries(
      providers.map(({ label, litellm_id, models, source }) => [label, { litellm_id, models, source: source ?? null }]),
    ),
  })
}

export async function saveModelTabOrder(order: string[]): Promise<ModelsResponse> {
  return put<ModelsResponse>("/models/tab-order", { order })
}

export async function fetchOmpCatalog(refresh = false): Promise<OmpCatalog> {
  return request<OmpCatalog>(`/models/omp${refresh ? "?refresh=true" : ""}`)
}

export async function saveModelDefaults(defaults: ModelDefaults): Promise<ModelsResponse> {
  return put<ModelsResponse>("/models/defaults", defaults)
}

export async function fetchPrompts(signal?: AbortSignal): Promise<Record<string, string>> {
  const data = await request<{ prompts: Record<string, string> }>("/prompts", { signal })
  return data.prompts
}

export async function fetchPromptList(): Promise<PromptMeta[]> {
  return request<PromptMeta[]>("/prompts/list")
}

export async function fetchPromptBlocks(): Promise<Record<string, string>> {
  return request<Record<string, string>>("/prompts/blocks")
}

export async function fetchPromptRaw(slug: string): Promise<{ slug: string; content: string }> {
  return request<{ slug: string; content: string }>(`/prompts/${encodeURIComponent(slug)}`)
}

export async function savePrompt(slug: string, content: string): Promise<{ slug: string; name: string }> {
  return put(`/prompts/${encodeURIComponent(slug)}`, { content })
}

export async function deletePrompt(slug: string): Promise<void> {
  await del(`/prompts/${encodeURIComponent(slug)}`)
}

export async function savePromptOrder(order: string[]): Promise<void> {
  await put("/prompt-order", { order })
}
