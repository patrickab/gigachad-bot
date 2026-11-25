// Memory-store routes: list a level, read a memory or document, apply a batch of ops.
import { toQuery } from "../urls"
import { post, request } from "./http"
import type { MemoryListing, MemoryOp, MemoryOpResult, MemoryRead } from "@/lib/domain/memory"

/** Top-level memory_stores of the root when `memoryStore` is empty, else one level of that memory_store. */
export async function listMemories(memoryStore = "", path = ""): Promise<MemoryListing> {
  return request<MemoryListing>(`/memory/list${toQuery({ memory_store: memoryStore || null, path: path || null })}`)
}

/** A memory's description and active facts, or a document's text when `path` ends in `.md`. */
export async function readMemory(memoryStore: string, path: string): Promise<MemoryRead> {
  return request<MemoryRead>(`/memory/read${toQuery({ memory_store: memoryStore, path })}`)
}

/** Apply `ops` in order; a failing op is reported in its result and the others still apply. */
export async function applyMemoryOps(memoryStore: string, ops: MemoryOp[]): Promise<MemoryOpResult[]> {
  const { results } = await post<{ results: MemoryOpResult[] }>("/memory/apply", { memory_store: memoryStore, ops })
  return results
}
