import type { MemoryTopic, StoredMemory } from "@/lib/domain/memory"

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

/** Apply `patch` to topic `name`, adding it when only memories reference it. */
export function patchTopic(topics: MemoryTopic[], name: string, patch: Partial<MemoryTopic>): MemoryTopic[] {
  if (!topics.some((t) => t.name === name)) return [...topics, { name, summary: "", ...patch }]
  return topics.map((t) => t.name === name ? { ...t, ...patch } : t)
}

const DAY_MS = 86_400_000

/** Relative "Updated ..." label of a timestamp, "Empty" without one. */
export function formatUpdated(iso: string | null): string {
  if (!iso) return "Empty"
  const date = new Date(iso)
  const now = new Date()
  const days = Math.round((new Date(now.toDateString()).getTime() - new Date(date.toDateString()).getTime()) / DAY_MS)
  if (days <= 0) return "Updated today"
  if (days === 1) return "Updated yesterday"
  if (days < 7) return `Updated ${days} days ago`
  const options: Intl.DateTimeFormatOptions = date.getFullYear() === now.getFullYear()
    ? { month: "short", day: "numeric" }
    : { month: "short", day: "numeric", year: "numeric" }
  return `Updated ${date.toLocaleDateString(undefined, options)}`
}
