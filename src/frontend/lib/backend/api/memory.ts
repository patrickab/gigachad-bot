// Memory extraction, review and storage routes.
import { toQuery } from "../urls"
import { post, put, request } from "./http"
import type { MemoryExtractResponse, MemoryOperation, MemoryScope, MemoryTopic, StoredMemory } from "@/lib/domain/memory"

/** Propose memory operations for the global scope and, with `projectSlug`, the project scope. */
export async function extractMemories(
  messages: { role: string; content: string }[],
  projectSlug?: string | null,
  chatId?: string | null,
): Promise<MemoryExtractResponse> {
  return post<MemoryExtractResponse>("/memory/extract", {
    messages,
    project_slug: projectSlug ?? null,
    chat_id: chatId ?? null,
  })
}

export async function cancelMemories(reviewId: string): Promise<void> {
  await post("/memory/cancel", { review_id: reviewId })
}

/** Apply accepted operations and topics of one scope, advancing the review's watermark. */
export async function commitMemories(
  scope: MemoryScope,
  operations: MemoryOperation[],
  topics: MemoryTopic[],
  projectSlug: string | null,
  reviewId: string,
): Promise<void> {
  await post("/memory/commit", { scope, operations, topics, project_slug: projectSlug, review_id: reviewId })
}

export async function getMemories(scope: MemoryScope, projectSlug?: string | null): Promise<{ memories: StoredMemory[] }> {
  return request(`/memory/memories${toQuery({ scope, project_slug: projectSlug })}`)
}

export async function saveMemories(scope: MemoryScope, memories: StoredMemory[], projectSlug?: string | null): Promise<void> {
  await put("/memory/memories", { scope, memories, project_slug: projectSlug ?? null })
}

export async function getTopics(scope: MemoryScope, projectSlug?: string | null, signal?: AbortSignal): Promise<{ topics: MemoryTopic[] }> {
  return request(`/memory/topics${toQuery({ scope, project_slug: projectSlug })}`, { signal })
}

export async function saveTopics(scope: MemoryScope, topics: MemoryTopic[], projectSlug?: string | null): Promise<void> {
  await put("/memory/topics", { scope, topics, project_slug: projectSlug ?? null })
}

/** Proposed regrouping of a scope's facts into subject topics, nothing is written. */
export async function regroupMemories(scope: MemoryScope, projectSlug?: string | null): Promise<{ topics: MemoryTopic[]; memories: StoredMemory[] }> {
  return post("/memory/regroup", { scope, project_slug: projectSlug ?? null })
}
