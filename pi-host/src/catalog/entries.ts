/**
 * `GET /sessions/:id/entries`: a session file's entries for a transcript shown without attaching (the picker preview).
 *
 * - Consumes: `Host.readEntries` and `Host.sessionMeta` (its `modifiedAt` keys the cache)
 * - Provides: `sessionEntries`, the whole file or one of two cuts of it:
 *   - with `turns`, a preview: the active branch from its `turns`-th last prompt on, tool result images left out
 *     (screenshots make up most of a long session file and a preview shows none)
 *   - with `customType` and `since`, a timeline: the `custom` entries of a type prefix, only those appended after an
 *     entry (the factory task view reads `factory-` entries this way, again from its last `leafId` after each change).
 *     `assistant` adds the assistant answers among them as `assistantAnswer` (text, model, usage; no tool calls or
 *     thinking), what the coder said in each round
 */
import type { Host, SessionEntries } from "../host/index.ts";
import { withMediaHashes } from "../media.ts";

/** Which cut of the file to answer: `turns` excludes the other three. */
export type EntriesQuery = { turns?: number; customType?: string; since?: string; assistant?: boolean };

/** Answers kept per session file version and query, so a preview shown again needs no file read. */
const CACHE_SIZE = 32;
const cache = new Map<string, { modifiedAt: string; query: string; answer: SessionEntries }>();

/** Drop a session file's cached answers (a rewrite must not leave removed entries in memory). */
export function forgetEntries(sessionPath: string): void {
  cache.delete(sessionPath);
}

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
 * An assistant message entry cut to what a timeline shows: `type: "message"`, its id and timestamp, and a message of
 * `role`, the joined `text` blocks, `model`, `stopReason` and `usage` (input with cache reads and writes, output,
 * `cost` in dollars). Undefined for any other entry.
 */
export function assistantAnswer(entry: Entry): Entry | undefined {
  const message = entry.message as Record<string, unknown> | undefined;
  if (entry.type !== "message" || message?.role !== "assistant") return undefined;
  const text = Array.isArray(message.content)
    ? message.content.filter((block) => block?.type === "text" && typeof block.text === "string").map((block) => block.text as string).join("\n\n")
    : "";
  const usage = message.usage as { input?: number; output?: number; cacheRead?: number; cacheWrite?: number; cost?: { total?: number } } | undefined;
  return {
    type: "message",
    id: entry.id,
    timestamp: entry.timestamp,
    message: {
      role: "assistant",
      text,
      model: message.model ?? null,
      stopReason: message.stopReason ?? null,
      usage: {
        input: (usage?.input ?? 0) + (usage?.cacheRead ?? 0) + (usage?.cacheWrite ?? 0),
        output: usage?.output ?? 0,
        cost: usage?.cost?.total ?? 0,
      },
    },
  };
}

/**
 * The entries appended after the entry `since` (all when undefined), only the `custom` ones whose `customType` starts
 * with `customType` when given, and with `assistant` the assistant answers among them (`assistantAnswer`). `leafId`
 * stays the file's last entry, the `since` of the next read. Undefined when no entry has the id `since`.
 */
export function entriesAfter({ entries, leafId }: SessionEntries, customType?: string, since?: string, assistant = false): SessionEntries | undefined {
  const start = since === undefined ? 0 : entries.findIndex((entry) => entry.id === since) + 1;
  if (since !== undefined && start === 0) return undefined;
  const later = entries.slice(start);
  if (customType === undefined) return { entries: later, leafId };
  const matching = later.flatMap((entry) => {
    if (entry.type === "custom" && typeof entry.customType === "string" && entry.customType.startsWith(customType)) return [entry];
    const answer = assistant ? assistantAnswer(entry) : undefined;
    return answer ? [answer] : [];
  });
  return { entries: matching, leafId };
}

/**
 * The entries of the session file at `sessionPath` cut as `query` says, cached per file version and query. Undefined
 * when `query.since` names no entry of the file.
 */
export async function sessionEntries(host: Host, sessionPath: string, query: EntriesQuery = {}): Promise<SessionEntries | undefined> {
  const { turns, customType, since, assistant } = query;
  const queryKey = JSON.stringify([turns, customType, since, assistant]);
  const modifiedAt = (await host.sessionMeta(sessionPath))?.modifiedAt ?? "";
  const cached = cache.get(sessionPath);
  if (cached && cached.modifiedAt === modifiedAt && cached.query === queryKey) return cached.answer;
  const all = await host.readEntries(sessionPath);
  const cut = turns === undefined ? entriesAfter(all, customType, since, assistant) : branchTail(all, turns);
  const answer = cut && withMediaHashes(cut);
  if (!answer) return undefined;
  cache.delete(sessionPath);
  cache.set(sessionPath, { modifiedAt, query: queryKey, answer });
  if (cache.size > CACHE_SIZE) cache.delete(cache.keys().next().value!);
  return answer;
}
