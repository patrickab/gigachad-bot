/**
 * `GET /sessions/:id/entries`: a session file's entries for a transcript shown without attaching (the picker preview).
 *
 * - Consumes: `Host.readEntries` and `Host.sessionMeta` (its `modifiedAt` keys the cache)
 * - Provides: `sessionEntries`, the whole file or one of two cuts of it:
 *   - with `turns`, a preview: the active branch from its `turns`-th last prompt on, tool result images left out
 *     (screenshots make up most of a long session file and a preview shows none)
 *   - with `customType` and `since`, a timeline: the `custom` entries of a type prefix, only those appended after an
 *     entry (the factory task view reads `factory-` entries this way, again from its last `leafId` after each change)
 */
import type { Host, SessionEntries } from "../host/index.ts";

/** Which cut of the file to answer: `turns` excludes the other two. */
export type EntriesQuery = { turns?: number; customType?: string; since?: string };

/** Answers kept per session file version and query, so a preview shown again needs no file read. */
const CACHE_SIZE = 32;
const cache = new Map<string, { modifiedAt: string; query: string; answer: SessionEntries }>();

type Entry = SessionEntries["entries"][number];

/** `entry` without the images of a tool result, the rest as it is. */
function withoutToolImages(entry: Entry): Entry {
  const message = entry.message as { role?: unknown; content?: unknown } | undefined;
  if (message?.role !== "toolResult" || !Array.isArray(message.content)) return entry;
  return { ...entry, message: { ...message, content: message.content.filter((block) => block?.type !== "image") } };
}

function isPrompt(entry: Entry): boolean {
  return entry.type === "message" && (entry.message as { role?: unknown } | undefined)?.role === "user";
}

/**
 * The active branch of `entries` (root to `leafId`) from its `turns`-th last prompt on, tool result images left out.
 *
 * The first kept entry's parent is left out, so a reader walking parents from the leaf stops there.
 */
export function branchTail({ entries, leafId }: SessionEntries, turns: number): SessionEntries {
  const byId = new Map(entries.map((entry) => [entry.id as string, entry]));
  const kept: Entry[] = [];
  let prompts = 0;
  let entry = leafId ? byId.get(leafId) : undefined;
  // Bounded by the entry count, so a broken parent chain cannot loop
  while (entry && kept.length < entries.length) {
    kept.push(withoutToolImages(entry));
    if (isPrompt(entry) && ++prompts === turns) break;
    entry = typeof entry.parentId === "string" ? byId.get(entry.parentId) : undefined;
  }
  return { entries: kept.reverse(), leafId };
}

/**
 * The entries appended after the entry `since` (all when undefined), only the `custom` ones whose `customType` starts
 * with `customType` when given. `leafId` stays the file's last entry, the `since` of the next read. Undefined when no
 * entry has the id `since`.
 */
export function entriesAfter({ entries, leafId }: SessionEntries, customType?: string, since?: string): SessionEntries | undefined {
  const start = since === undefined ? 0 : entries.findIndex((entry) => entry.id === since) + 1;
  if (since !== undefined && start === 0) return undefined;
  const later = entries.slice(start);
  if (customType === undefined) return { entries: later, leafId };
  const matching = later.filter(
    (entry) => entry.type === "custom" && typeof entry.customType === "string" && entry.customType.startsWith(customType),
  );
  return { entries: matching, leafId };
}

/**
 * The entries of the session file at `sessionPath` cut as `query` says, cached per file version and query. Undefined
 * when `query.since` names no entry of the file.
 */
export async function sessionEntries(host: Host, sessionPath: string, query: EntriesQuery = {}): Promise<SessionEntries | undefined> {
  const { turns, customType, since } = query;
  const queryKey = JSON.stringify([turns, customType, since]);
  const modifiedAt = (await host.sessionMeta(sessionPath))?.modifiedAt ?? "";
  const cached = cache.get(sessionPath);
  if (cached && cached.modifiedAt === modifiedAt && cached.query === queryKey) return cached.answer;
  const all = await host.readEntries(sessionPath);
  const answer = turns === undefined ? entriesAfter(all, customType, since) : branchTail(all, turns);
  if (!answer) return undefined;
  cache.delete(sessionPath);
  cache.set(sessionPath, { modifiedAt, query: queryKey, answer });
  if (cache.size > CACHE_SIZE) cache.delete(cache.keys().next().value!);
  return answer;
}
