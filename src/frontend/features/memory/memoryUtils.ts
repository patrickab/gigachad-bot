import type { Fact } from "@/lib/domain/memory"
import { relativeIn } from "@/lib/paths"

/** A level of the memory window: the root when `memoryStore` is empty, else a memory_store or one of its directories. */
export interface MemoryLocation {
  memoryStore: string
  path: string
}

export const ROOT_LOCATION: MemoryLocation = { memoryStore: "", path: "" }

/** Breadcrumb label of a location: `gigachad-bot/memory-store-v1 / map`, `Memory` for the root. */
export function locationLabel(location: MemoryLocation): string {
  if (!location.memoryStore) return "Memory"
  return location.path ? `${location.memoryStore} / ${location.path}` : location.memoryStore
}

/** Breadcrumb label of `child` opened from `parent`, without the part `parent` already names. */
export function childLabel(parent: MemoryLocation, child: MemoryLocation): string {
  if (child.memoryStore === parent.memoryStore) return relativeIn(parent.path, child.path)
  if (parent.memoryStore && !child.path) return relativeIn(parent.memoryStore, child.memoryStore)
  return locationLabel(child)
}

/** Active facts newest `updated` first; ties keep their order. */
export function newestFirst(facts: Fact[]): Fact[] {
  return [...facts].sort((a, b) => b.updated.localeCompare(a.updated))
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
