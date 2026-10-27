import type { MemoryTopic, StoredMemory } from "@/lib/domain/types"

/** Backend fallback topic of a fact without one. */
const FALLBACK_TOPIC = "note"

/** Display title of a stored topic name: `dev_environment` -> `Dev Environment`. */
export function formatTopicHeading(name: string): string {
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
}

/** Stored topic name of a typed title, mirrors the backend `topic_name`: `Dev Environment` -> `dev_environment`. */
export function topicNameFromTitle(title: string): string {
  return title.trim().toLowerCase().replace(/[^\p{L}\p{N}_]+/gu, "_").replace(/^_+|_+$/g, "")
}

export function normalizeMemoryTopic(memory: StoredMemory): string {
  return memory.topic || FALLBACK_TOPIC
}

/** Locally unique id for a manually added memory. */
export function newManualMemoryId(): string {
  return `manual-${Date.now()}-${Math.random().toString(36).slice(2)}`
}

/** Group memories by topic in stored topic order (empty topics included), then topics only facts reference. */
export function groupByTopicOrder(topics: MemoryTopic[], memories: StoredMemory[]): { topic: MemoryTopic; items: StoredMemory[] }[] {
  const byTopic = new Map<string, StoredMemory[]>()
  for (const memory of memories) {
    const name = normalizeMemoryTopic(memory)
    byTopic.set(name, [...(byTopic.get(name) ?? []), memory])
  }
  const known = new Set(topics.map((t) => t.name))
  return [
    ...topics.map((topic) => ({ topic, items: byTopic.get(topic.name) ?? [] })),
    ...[...byTopic].filter(([name]) => !known.has(name)).map(([name, items]) => ({ topic: { name, summary: "" }, items })),
  ]
}
