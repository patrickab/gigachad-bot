/**
 * Removing a turn from a session file's entries (conversation unification, wave pi/1).
 *
 * - Consumes: the parsed header and entries of a session file, no I/O
 * - Provides: `removeTurn`, the entries without the turn and the removed entries (their media and pastes go next)
 *
 * The active path ends at the last entry in the file, as in Pi (`SessionManager._buildIndex`). A turn is a user prompt
 * and the entries after it on that path up to, not including, the next user prompt or compaction. Pi reads the leaf
 * from the last line, so the new leaf is moved to the end when it is not there: Pi builds its tree from `parentId`
 * alone, the order of the other lines does not matter to it.
 */

import { isPrompt, pathToRoot, type Entry } from "../entries.ts";

export type TurnEntry = Entry;

/** `turn` removes one turn, `end` the turn and everything after it (an edit). */
export type RemoveThrough = "turn" | "end";

/**
 * Why a turn cannot be removed:
 * - `unknown`: no entry has the id
 * - `not_prompt`: the entry is no user message
 * - `off_branch`: the prompt is not on the active path, so "the entries after it" are not defined
 * - `compacted`: a compaction that stays summarized the turn, its content lives on inside the summary
 */
export type RemoveTurnRefusal = "unknown" | "not_prompt" | "off_branch" | "compacted";

export type RemoveTurnResult =
  | { ok: true; entries: TurnEntry[]; removed: TurnEntry[]; leafId: string | null }
  | { ok: false; reason: RemoveTurnRefusal };

/** Kept and re-linked when a `turn` range holds them: they set the session's state, not the turn's content. */
const STATE_TYPES = new Set(["model_change", "thinking_level_change", "session_info"]);
/** Entries that point at another one by `targetId`: they go when their target goes. */
const TARGETING_TYPES = new Set(["label", "context_edit"]);

/**
 * `entries` without the turn of `promptEntryId` (`through: "turn"`) or without it and everything after it
 * (`through: "end"`, the entry before the prompt becomes the leaf).
 *
 * - Side branches rooted in the range go with it, the first surviving entry after it takes the range's parent
 * - `model_change`, `thinking_level_change` and `session_info` entries in a `turn` range stay, re-linked; so do
 *   `label` and `context_edit` entries there whose target stays. Every other entry of the range goes
 * - `label` and `context_edit` entries whose target is gone go too
 * - A `compaction.firstKeptEntryId` in the range moves to the next surviving entry on the path
 */
export function removeTurn(entries: TurnEntry[], promptEntryId: string, through: RemoveThrough): RemoveTurnResult {
  const byId = new Map(entries.map((entry) => [entry.id, entry]));
  const prompt = byId.get(promptEntryId);
  if (!prompt) return { ok: false, reason: "unknown" };
  if (!isPrompt(prompt)) return { ok: false, reason: "not_prompt" };

  const path = pathToRoot(entries, entries.at(-1)?.id ?? null);
  const start = path.indexOf(prompt);
  if (start < 0) return { ok: false, reason: "off_branch" };

  const children = new Map<string, TurnEntry[]>();
  for (const entry of entries) {
    if (entry.parentId === null) continue;
    (children.get(entry.parentId) ?? children.set(entry.parentId, []).get(entry.parentId)!).push(entry);
  }
  const removed = new Set<string>();
  const removeSubtree = (root: TurnEntry) => {
    const stack = [root];
    for (let entry = stack.pop(); entry; entry = stack.pop()) {
      if (removed.has(entry.id)) continue;
      removed.add(entry.id);
      stack.push(...(children.get(entry.id) ?? []));
    }
  };

  if (through === "end") {
    removeSubtree(prompt);
  } else {
    let end = start + 1;
    while (end < path.length && !isPrompt(path[end]!) && path[end]!.type !== "compaction") end++;
    const range = path.slice(start, end);
    const onPath = new Set(path.map((entry) => entry.id));
    for (const entry of range) {
      for (const child of children.get(entry.id) ?? []) if (!onPath.has(child.id)) removeSubtree(child);
      if (!STATE_TYPES.has(entry.type) && !TARGETING_TYPES.has(entry.type)) removed.add(entry.id);
    }
  }

  // A compaction that stays and keeps entries from after the prompt on summarized the prompt
  const position = new Map(path.map((entry, index) => [entry.id, index]));
  for (const entry of path.slice(start + 1)) {
    if (entry.type !== "compaction" || removed.has(entry.id)) continue;
    const firstKept = position.get(entry.firstKeptEntryId as string);
    if (firstKept !== undefined && firstKept > start) return { ok: false, reason: "compacted" };
  }

  // Targets can chain (a label of a label), so repeat until nothing more goes
  for (let changed = true; changed; ) {
    changed = false;
    for (const entry of entries) {
      if (removed.has(entry.id) || !TARGETING_TYPES.has(entry.type)) continue;
      const target = entry.targetId;
      if (typeof target === "string" && (removed.has(target) || !byId.has(target))) {
        removed.add(entry.id);
        changed = true;
      }
    }
  }

  const survivor = (id: string | null): string | null => {
    for (let current = id; current !== null; current = byId.get(current)?.parentId ?? null) {
      if (!removed.has(current)) return current;
    }
    return null;
  };
  /** The first surviving entry on the path at or after `id`, the compaction itself when none comes before it. */
  const nextOnPath = (id: string, compaction: TurnEntry): string => {
    for (const entry of path.slice(position.get(id) ?? path.length)) if (!removed.has(entry.id)) return entry.id;
    return compaction.id;
  };

  const kept: TurnEntry[] = [];
  for (const entry of entries) {
    if (removed.has(entry.id)) continue;
    let next = entry;
    if (entry.parentId !== null && removed.has(entry.parentId)) next = { ...next, parentId: survivor(entry.parentId) };
    if (entry.type === "compaction" && typeof entry.firstKeptEntryId === "string" && removed.has(entry.firstKeptEntryId)) {
      next = { ...next, firstKeptEntryId: nextOnPath(entry.firstKeptEntryId, entry) };
    }
    kept.push(next);
  }

  const leafId = survivor(through === "end" ? prompt.parentId : path.at(-1)!.id);
  const leafIndex = kept.findIndex((entry) => entry.id === leafId);
  if (leafIndex >= 0 && leafIndex !== kept.length - 1) kept.push(...kept.splice(leafIndex, 1));
  return { ok: true, entries: kept, removed: entries.filter((entry) => removed.has(entry.id)), leafId };
}
