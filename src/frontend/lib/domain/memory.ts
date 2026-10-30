export type MemoryScope = "global" | "project"

/** Atomic memory: one stored fact within a topic. */
export interface StoredMemory {
  id: string
  text: string
  topic: string
  scope: MemoryScope
  created_at?: string
  updated_at?: string
}

/** Complete memory: a named subject bundling atomic memories. */
export interface MemoryTopic {
  name: string
  summary: string
}

/** Proposed change to one fact, `target_*` name the stored fact an update or delete replaces. `index` is the 0-based place in the relevance order, null appends an add and keeps an update in place. */
export interface MemoryOperation {
  id: string
  op: "add" | "update" | "delete"
  scope: MemoryScope
  topic: string
  text: string
  target_id: string | null
  target_text: string | null
  index: number | null
}

/** New topic (`previous_summary` null) or a revised summary of an existing one. */
export interface TopicProposal {
  scope: MemoryScope
  name: string
  summary: string
  previous_summary: string | null
}

export interface ScopeProposal {
  operations: MemoryOperation[]
  topics: TopicProposal[]
}

/** A scope is null when it was not extracted. */
export interface MemoryExtractResponse {
  review_id: string
  global: ScopeProposal | null
  project: ScopeProposal | null
}
