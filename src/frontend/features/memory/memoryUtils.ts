import type { MemoryTopic, PreviewMemory, ProposedMemory } from "@/lib/domain/types"

export type MemoryRecord = ProposedMemory | PreviewMemory

type MemorySection = {
  topic: MemoryTopic | { name: string; summary: string }
  items: MemoryRecord[]
}

export function formatTopicName(name: string): string {
  return name.replace(/_/g, " ")
}

/** Display title of a stored topic name: `dev_environment` -> `Dev Environment`. */
export function formatTopicHeading(name: string): string {
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
}

/** Stored topic name of a typed title: `Dev Environment` -> `dev_environment`. */
export function topicNameFromTitle(title: string): string {
  return title.trim().toLowerCase().replace(/\s+/g, "_")
}

/** Missing or empty topics group under "note", the backend fallback. */
export function normalizeMemoryTopic(m: MemoryRecord): string {
  return m.topic || "note"
}

export function memoryText(m: MemoryRecord): string {
  return "memory" in m ? m.memory : m.text
}

/** Locally unique id for a manually added memory. */
export function newManualMemoryId(): string {
  return `manual-${Date.now()}-${Math.random().toString(36).slice(2)}`
}

// Newest first by updated_at, falling back to created_at. ISO timestamps compare lexically.
function sortByUpdated(mems: MemoryRecord[]): MemoryRecord[] {
  return [...mems].sort((a, b) => {
    const ta = ("updated_at" in a ? a.updated_at : undefined) ?? ("created_at" in a ? a.created_at : undefined) ?? ""
    const tb = ("updated_at" in b ? b.updated_at : undefined) ?? ("created_at" in b ? b.created_at : undefined) ?? ""
    return tb.localeCompare(ta)
  })
}

/** Group memories into sections in definition order (empty topics included), then any unknown topics last. */
export function groupByTopicOrder(
  topicOrder: MemoryTopic[],
  memories: MemoryRecord[],
): MemorySection[] {
  const byTopic = new Map<string, MemoryRecord[]>()
  for (const m of memories) {
    const cat = normalizeMemoryTopic(m)
    const arr = byTopic.get(cat) ?? []
    arr.push(m)
    byTopic.set(cat, arr)
  }

  const sections: MemorySection[] = []
  const seen = new Set<string>()

  for (const cat of topicOrder) {
    seen.add(cat.name)
    sections.push({ topic: cat, items: sortByUpdated(byTopic.get(cat.name) ?? []) })
  }

  for (const [name, items] of byTopic) {
    if (!seen.has(name)) {
      sections.push({ topic: { name, summary: "" }, items: sortByUpdated(items) })
    }
  }

  return sections
}

/** Mirror the backend `render_memories_as_markdown` so live edits update the diff without a round-trip. */
export function renderMemoriesMarkdown(
  memories: PreviewMemory[],
  title: string,
  topicOrder?: MemoryTopic[],
): string {
  const parts = [`# ${title}`]
  for (const { topic, items } of groupByTopicOrder(topicOrder ?? [], memories)) {
    if (items.length === 0) continue
    parts.push(`\n## ${formatTopicHeading(topic.name)}`)
    for (const m of items) parts.push(`- ${memoryText(m)}`)
  }
  return parts.join("\n")
}
