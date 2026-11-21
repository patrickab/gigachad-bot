/**
 * `GET /sessions/:id/entries`: a session file's entries for a transcript shown without attaching (the picker preview).
 *
 * - Consumes: `Host.readEntries` and `Host.sessionMeta` (its `modifiedAt` keys the cache)
 * - Provides: `sessionEntries`, the whole file or, with `turns`, a preview: the active branch from its `turns`-th last
 *   prompt on, tool result images left out (screenshots make up most of a long session file and a preview shows none)
 */
import type { Host, SessionEntries } from "../host/index.ts";

/** Answers kept per session file version, so a preview shown again needs no file read. */
const CACHE_SIZE = 32;
const cache = new Map<string, { modifiedAt: string; turns: number | undefined; answer: SessionEntries }>();

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

/** The entries of the session file at `sessionPath`, its last `turns` turns when given; cached per file version. */
export async function sessionEntries(host: Host, sessionPath: string, turns: number | undefined): Promise<SessionEntries> {
  const modifiedAt = (await host.sessionMeta(sessionPath))?.modifiedAt ?? "";
  const cached = cache.get(sessionPath);
  if (cached && cached.modifiedAt === modifiedAt && cached.turns === turns) return cached.answer;
  const all = await host.readEntries(sessionPath);
  const answer = turns === undefined ? all : branchTail(all, turns);
  cache.delete(sessionPath);
  cache.set(sessionPath, { modifiedAt, turns, answer });
  if (cache.size > CACHE_SIZE) cache.delete(cache.keys().next().value!);
  return answer;
}
