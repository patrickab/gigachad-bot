import type { CategoryDef, PreviewMemory, ProposedMemory } from "@/lib/domain/types"

export type MemoryRecord = ProposedMemory | PreviewMemory

type MemorySection = {
  category: CategoryDef | { name: string; description: string }
  items: MemoryRecord[]
}

export function formatCategoryName(name: string): string {
  return name.replace(/_/g, " ")
}

function formatCategoryHeading(name: string): string {
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
}

/** Missing or empty categories group under "uncategorized". */
export function normalizeMemoryCategory(m: MemoryRecord): string {
  return m.category || "uncategorized"
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

/** Group memories into sections in definition order (empty categories included), then any unknown categories last. */
export function groupByCategoryOrder(
  categoryOrder: CategoryDef[],
  memories: MemoryRecord[],
): MemorySection[] {
  const byCategory = new Map<string, MemoryRecord[]>()
  for (const m of memories) {
    const cat = normalizeMemoryCategory(m)
    const arr = byCategory.get(cat) ?? []
    arr.push(m)
    byCategory.set(cat, arr)
  }

  const sections: MemorySection[] = []
  const seen = new Set<string>()

  for (const cat of categoryOrder) {
    seen.add(cat.name)
    sections.push({ category: cat, items: sortByUpdated(byCategory.get(cat.name) ?? []) })
  }

  for (const [name, items] of byCategory) {
    if (!seen.has(name)) {
      sections.push({ category: { name, description: "" }, items: sortByUpdated(items) })
    }
  }

  return sections
}

/** Flatten sections back into a category-ordered memory list. */
export function sortMemoriesByCategoryOrder(mems: PreviewMemory[], categoryOrder: CategoryDef[]): PreviewMemory[] {
  return groupByCategoryOrder(categoryOrder, mems).flatMap((s) => s.items) as PreviewMemory[]
}

/** Mirror the backend `render_memories_as_markdown` so live edits update the diff without a round-trip. */
export function renderMemoriesMarkdown(
  memories: PreviewMemory[],
  title: string,
  categoryOrder?: CategoryDef[],
): string {
  const parts = [`# ${title}`]
  for (const { category, items } of groupByCategoryOrder(categoryOrder ?? [], memories)) {
    if (items.length === 0) continue
    parts.push(`\n## ${formatCategoryHeading(category.name)}`)
    for (const m of items) parts.push(`- ${memoryText(m)}`)
  }
  return parts.join("\n")
}
