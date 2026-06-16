import type { CategoryDef, PreviewMemory, ProposedMemory } from "@/lib/types"

type MemoryRecord = ProposedMemory | PreviewMemory

export type MemorySection = {
  category: CategoryDef | { name: string; description: string }
  items: MemoryRecord[]
}

export function formatCategoryName(name: string): string {
  return name.replace(/_/g, " ")
}

export function formatCategoryHeading(name: string): string {
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
}

export function normalizeMemoryCategory(m: MemoryRecord): string {
  return m.category || "uncategorized"
}

function sortByUpdated(mems: MemoryRecord[]): MemoryRecord[] {
  return [...mems].sort((a, b) => {
    const ta = ("updated_at" in a ? a.updated_at : undefined) ?? ("created_at" in a ? a.created_at : undefined) ?? ""
    const tb = ("updated_at" in b ? b.updated_at : undefined) ?? ("created_at" in b ? b.created_at : undefined) ?? ""
    return tb.localeCompare(ta)
  })
}

/** Group memories into category sections in definition order (includes empty categories). */
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

export const buildBoardSections = groupByCategoryOrder

/** Flatten sections back into a category-ordered memory list. */
export function sortMemoriesByCategoryOrder(mems: PreviewMemory[], categoryOrder: CategoryDef[]): PreviewMemory[] {
  return groupByCategoryOrder(categoryOrder, mems).flatMap((s) => s.items) as PreviewMemory[]
}
