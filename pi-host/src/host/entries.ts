/**
 * Session file entries: the entry shape, user prompts and the branch walk by `parentId`.
 *
 * - Consumes: parsed session entries, no I/O
 * - Provides: `Entry`, `isPrompt`, `pathToRoot`
 */

export type Entry = { type: string; id: string; parentId: string | null; [field: string]: unknown };

/** True for a user message entry. */
export function isPrompt(entry: { type?: unknown; message?: unknown }): boolean {
  return entry.type === "message" && (entry.message as { role?: unknown } | undefined)?.role === "user";
}

/** Root-to-leaf path by `parentId`; a missing parent or a repeated id ends the walk. */
export function pathToRoot<T extends { id: string; parentId: string | null }>(entries: readonly T[], leafId: string | null): T[] {
  const byId = new Map(entries.map((entry) => [entry.id, entry]));
  const path: T[] = [];
  const seen = new Set<string>();
  for (let entry = leafId === null ? undefined : byId.get(leafId); entry && !seen.has(entry.id); ) {
    seen.add(entry.id);
    path.push(entry);
    entry = entry.parentId === null ? undefined : byId.get(entry.parentId);
  }
  return path.reverse();
}
