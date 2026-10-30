import { groupByTopicOrder, newManualMemoryId, normalizeMemoryTopic, patchTopic } from "@/features/memory/memoryUtils"
import type { MemoryScope, MemoryTopic, StoredMemory } from "@/lib/domain/memory"

/** One memory scope: the global profile ("You") or one project. */
export interface ScopeMemory {
  key: string
  scope: MemoryScope
  projectSlug: string | null
  label: string
  topics: MemoryTopic[]
  memories: StoredMemory[]
}

export type ScopeContent = Pick<ScopeMemory, "topics" | "memories">
export type ScopeChange = (scope: ScopeMemory) => Partial<ScopeContent>

export interface TopicRow {
  topic: MemoryTopic
  memories: StoredMemory[]
  updatedAt: string | null
}

function latestUpdate(memories: StoredMemory[]): string | null {
  return memories.reduce<string | null>((latest, m) => {
    const stamp = m.updated_at ?? m.created_at ?? null
    return stamp && (!latest || stamp > latest) ? stamp : latest
  }, null)
}

/** Topics in stored order plus topics only referenced by memories, each with its memories. */
export function topicRows(content: ScopeContent): TopicRow[] {
  return groupByTopicOrder(content.topics, content.memories).map(({ topic, items }) => ({ topic, memories: items, updatedAt: latestUpdate(items) }))
}

export function topicNames(scope: ScopeMemory): string[] {
  return topicRows(scope).map((row) => row.topic.name)
}

/** Point every memory of topic `name` at topic `next`. */
function retopic(memories: StoredMemory[], name: string, next: string): StoredMemory[] {
  return memories.map((m) => normalizeMemoryTopic(m) === name ? { ...m, topic: next } : m)
}

export function createTopic(name: string): ScopeChange {
  return (s) => ({ topics: [...s.topics, { name, summary: "" }] })
}

export function renameTopic(name: string, next: string): ScopeChange {
  return (s) => ({ topics: patchTopic(s.topics, name, { name: next }), memories: retopic(s.memories, name, next) })
}

export function mergeTopic(name: string, target: string): ScopeChange {
  return (s) => ({ topics: patchTopic(s.topics.filter((t) => t.name !== name), target, {}), memories: retopic(s.memories, name, target) })
}

export function setTopicSummary(name: string, summary: string): ScopeChange {
  return (s) => ({ topics: patchTopic(s.topics, name, { summary }) })
}

/** Add a manual memory with text `text` to topic `topic` of a scope of kind `scope`. */
export function addMemory(topic: string, text: string, scope: MemoryScope): ScopeChange {
  const now = new Date().toISOString()
  const memory: StoredMemory = { id: newManualMemoryId(), text, topic, scope, created_at: now, updated_at: now }
  return (s) => ({ topics: patchTopic(s.topics, topic, {}), memories: [...s.memories, memory] })
}

export function editMemory(id: string, text: string): ScopeChange {
  return (s) => ({ memories: s.memories.map((m) => m.id === id ? { ...m, text, updated_at: new Date().toISOString() } : m) })
}

export function moveMemory(id: string, topic: string): ScopeChange {
  return (s) => ({ topics: patchTopic(s.topics, topic, {}), memories: s.memories.map((m) => m.id === id ? { ...m, topic } : m) })
}

/** Swap a memory with its neighbour in the scope's relevance order, which is the stored order. */
export function shiftMemory(id: string, offset: -1 | 1): ScopeChange {
  return (s) => {
    const from = s.memories.findIndex((m) => m.id === id)
    const to = from + offset
    if (from < 0 || to < 0 || to >= s.memories.length) return {}
    const memories = [...s.memories]
    ;[memories[from], memories[to]] = [memories[to], memories[from]]
    return { memories }
  }
}

export function deleteTopic(name: string): ScopeChange {
  return (s) => ({ topics: s.topics.filter((t) => t.name !== name), memories: s.memories.filter((m) => normalizeMemoryTopic(m) !== name) })
}

/** Put a deleted topic back at its old index together with its memories. */
export function restoreTopic(index: number, topic: MemoryTopic | undefined, memories: StoredMemory[]): ScopeChange {
  return (s) => {
    const topics = [...s.topics]
    if (topic) topics.splice(Math.min(index, topics.length), 0, topic)
    return { topics, memories: [...s.memories, ...memories] }
  }
}

export function deleteMemory(id: string): ScopeChange {
  return (s) => ({ memories: s.memories.filter((m) => m.id !== id) })
}

/** Put a deleted memory back at its old index. */
export function restoreMemory(index: number, memory: StoredMemory): ScopeChange {
  return (s) => {
    const memories = [...s.memories]
    memories.splice(Math.min(index, memories.length), 0, memory)
    return { memories }
  }
}
