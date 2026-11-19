/**
 * Full-text search over this machine's live session transcripts (M24).
 *
 * - Consumes: session metadata from `sessions.ts`, the session files, scoring from `query.ts`
 * - Provides: `searchSessions(query, limit, sessions)` behind `LocalHost.searchSessions`
 *
 * Only the active branch's user and assistant text is searched: tool output and thinking would drown real matches.
 * Each file's extracted text is cached by size and mtime, so an unchanged file is never parsed twice; files are
 * parsed 4 at a time.
 *
 * Ported from omp-web `lib/session-content-search.ts` (MIT, Copyright (c) 2026 agegr,
 * https://github.com/ompweb-dev/ompweb), adapted to Pi's records and to report the hit's entry and a snippet.
 */
import { stat } from "node:fs/promises";
import type { SearchMatch, SessionMeta } from "../index.ts";
import {
  compileSearchQuery,
  CONTENT_WEIGHT,
  findSearchMatches,
  TITLE_WEIGHT,
  scoreSearchFields,
  searchField,
  type CompiledSearchQuery,
  type SearchField,
} from "./query.ts";

const PARSE_CONCURRENCY = 4;
const MAX_QUERY_CHARS = 500;
/** Lines longer than this skip `JSON.parse` when their head shows a record without searchable text. */
const LARGE_LINE_CHARS = 64 * 1024;
const LINE_HEAD_CHARS = 512;
const SNIPPET_BEFORE_CHARS = 60;
const SNIPPET_AFTER_CHARS = 140;

/** One record of a session file, reduced to what the active branch and the search need. */
type SessionRecord = {
  id: string;
  parentId: string | null;
  type: string;
  texts?: string[];
  firstKeptEntryId?: string;
  /** Session name of a `session_info` record. */
  name?: string;
};

/** The active branch's user and assistant texts, one per entry, and the searched fields (name and whole text). */
type Transcript = { entries: { id: string; text: string }[]; fields: SearchField[] };

// Holds every live session's extracted text in memory (a small share of the file size) and re-reads a whole file
// when it changes; move to an on-disk index if installs outgrow that.
const cache = new Map<string, { version: string; transcript: Promise<Transcript> }>();

/** Texts of a message's content (string or text blocks; thinking and tool calls are skipped). */
function textBlocks(content: unknown): string[] {
  if (typeof content === "string") return [content];
  if (!Array.isArray(content)) return [];
  return content.filter((block) => block?.type === "text" && typeof block.text === "string").map((block) => block.text as string);
}

/**
 * Head fields of a large line whose record has no searchable text (a tool result, a custom entry), undefined when
 * it may have some or its head does not read as Pi writes it: such lines are parsed whole.
 */
function skimLargeLine(line: string): Omit<SessionRecord, "texts"> | undefined {
  const head = line.slice(0, LINE_HEAD_CHARS);
  const type = /^\{"type":"([^"]+)"/.exec(head)?.[1];
  if (!type || type === "session" || type === "compaction") return undefined;
  const ids = /"id":"([^"]+)","parentId":(?:null|"([^"]+)")/.exec(head);
  if (!ids) return undefined;
  if (type === "message") {
    const role = /"message":\{"role":"([^"]+)"/.exec(head)?.[1];
    if (!role || role === "user" || role === "assistant") return undefined;
  }
  return { type, id: ids[1]!, parentId: ids[2] ?? null };
}

/** The search record of one line, undefined for the header, unparseable lines and records without an id. */
function parseRecord(line: string, previous: SessionRecord | undefined, index: number, legacy: boolean): SessionRecord | undefined {
  const skimmed = !legacy && line.length > LARGE_LINE_CHARS ? skimLargeLine(line) : undefined;
  if (skimmed) return skimmed;
  let raw: { [key: string]: unknown };
  try {
    raw = JSON.parse(line);
  } catch {
    return undefined;
  }
  if (typeof raw?.type !== "string" || raw.type === "session") return undefined;
  // Files without entry ids (version 1) chain each record to the one before.
  const id = legacy ? `legacy:${index}` : raw.id;
  const parentId = legacy ? (previous?.id ?? null) : raw.parentId;
  if (typeof id !== "string" || (parentId !== null && typeof parentId !== "string")) return undefined;
  const message = typeof raw.message === "object" && raw.message !== null ? (raw.message as { role?: unknown; content?: unknown }) : undefined;
  const searchable = message?.role === "user" || message?.role === "assistant";
  return {
    id,
    parentId: (parentId as string | null) ?? null,
    type: raw.type,
    ...(searchable ? { texts: textBlocks(message.content) } : {}),
    ...(typeof raw.firstKeptEntryId === "string" ? { firstKeptEntryId: raw.firstKeptEntryId } : {}),
    ...(raw.type === "session_info" && typeof raw.name === "string" ? { name: raw.name } : {}),
  };
}

/** Records on the path from the last record to the root, minus what the latest compaction summarized away. */
function activeBranch(records: SessionRecord[]): SessionRecord[] {
  const byId = new Map(records.map((record) => [record.id, record]));
  const path: SessionRecord[] = [];
  const seen = new Set<string>();
  let current = records.at(-1);
  while (current && !seen.has(current.id)) {
    seen.add(current.id);
    path.push(current);
    current = current.parentId ? byId.get(current.parentId) : undefined;
  }
  path.reverse();
  const compaction = path.findLastIndex((record) => record.type === "compaction");
  if (compaction < 0) return path;
  const firstKept = path.findIndex((record, index) => index < compaction && record.id === path[compaction]!.firstKeptEntryId);
  return [...path.slice(firstKept >= 0 ? firstKept : compaction, compaction), ...path.slice(compaction + 1)];
}

/** Stream a session file and extract its active branch's user and assistant text. */
async function readTranscript(sessionPath: string): Promise<Transcript> {
  const records: SessionRecord[] = [];
  const decoder = new TextDecoder();
  let pending = "";
  let index = 0;
  let legacy = false;
  let headerSeen = false;
  const consume = (line: string): void => {
    if (!line) return;
    if (!headerSeen) {
      headerSeen = true;
      try {
        const version = JSON.parse(line)?.version;
        legacy = version === undefined || version === null || version === 1;
      } catch {
        // Not a header: the file is not a Pi session and yields no records worth searching.
      }
      return;
    }
    const record = parseRecord(line, records.at(-1), index++, legacy);
    if (record) records.push(record);
  };
  for await (const chunk of Bun.file(sessionPath).stream()) {
    const lines = (pending + decoder.decode(chunk, { stream: true })).split("\n");
    pending = lines.pop() ?? "";
    for (const line of lines) consume(line);
  }
  consume(pending + decoder.decode());
  const entries = activeBranch(records).flatMap((record) => {
    const text = (record.texts ?? []).join("\n").trim();
    return text ? [{ id: record.id.startsWith("legacy:") ? "" : record.id, text }] : [];
  });
  const name = records.findLast((record) => record.name !== undefined)?.name;
  const content = searchField(entries.map((entry) => entry.text).join("\n"), CONTENT_WEIGHT);
  return { entries, fields: name ? [searchField(name, TITLE_WEIGHT), content] : [content] };
}

/** Transcript of a session file, reused while its size and mtime stay the same; undefined when unreadable. */
async function transcriptOf(sessionPath: string): Promise<Transcript | undefined> {
  let stats;
  try {
    stats = await stat(sessionPath);
  } catch {
    cache.delete(sessionPath);
    return undefined;
  }
  const version = `${stats.size}:${stats.mtimeMs}`;
  let cached = cache.get(sessionPath);
  if (cached?.version !== version) {
    const entry = { version, transcript: readTranscript(sessionPath) };
    cached = entry;
    cache.set(sessionPath, entry);
    // A failed read must not stay cached.
    entry.transcript.catch(() => {
      if (cache.get(sessionPath) === entry) cache.delete(sessionPath);
    });
  }
  return cached.transcript.catch(() => undefined);
}

/** Text around the entry's strongest hit, whitespace collapsed, `…` where it is cut. */
function snippetOf(text: string, query: CompiledSearchQuery): string {
  const spans = findSearchMatches(text, query);
  const best = spans.reduce<(typeof spans)[number] | undefined>((top, span) => (top && top.quality >= span.quality ? top : span), undefined);
  const start = best ? Math.max(0, best.start - SNIPPET_BEFORE_CHARS) : 0;
  const end = Math.min(text.length, (best?.end ?? 0) + SNIPPET_AFTER_CHARS);
  const body = text.slice(start, end).replace(/\s+/g, " ").trim();
  return `${start > 0 ? "…" : ""}${body}${end < text.length ? "…" : ""}`;
}

/** The entry with the most and best hits (the first on ties), with its snippet. */
function bestEntry(transcript: Transcript, query: CompiledSearchQuery): { entryId: string; snippet: string } {
  let best: { id: string; text: string } | undefined;
  let bestQuality = 0;
  for (const entry of transcript.entries) {
    const quality = findSearchMatches(entry.text, query).reduce((sum, span) => sum + span.quality, 0);
    if (quality > bestQuality) {
      best = entry;
      bestQuality = quality;
    }
  }
  const entry = best ?? transcript.entries[0];
  return entry ? { entryId: entry.id, snippet: snippetOf(entry.text, query) } : { entryId: "", snippet: "" };
}

/** `sessions` (newest first) matching `rawQuery`, best score first and newest first among equal scores. */
export async function searchSessions(rawQuery: string, limit: number, sessions: SessionMeta[]): Promise<SearchMatch[]> {
  const query = compileSearchQuery(rawQuery.slice(0, MAX_QUERY_CHARS));
  if (!query) return [];
  const listed = new Set(sessions.map((session) => session.sessionPath));
  for (const path of cache.keys()) if (!listed.has(path)) cache.delete(path);

  const hits: { match: SearchMatch; modifiedAt: string }[] = [];
  for (let start = 0; start < sessions.length; start += PARSE_CONCURRENCY) {
    await Promise.all(
      sessions.slice(start, start + PARSE_CONCURRENCY).map(async (session) => {
        const transcript = await transcriptOf(session.sessionPath);
        if (!transcript) return;
        const score = scoreSearchFields(transcript.fields, query);
        if (score === null) return;
        const match = { sessionPath: session.sessionPath, ...bestEntry(transcript, query), score };
        hits.push({ match, modifiedAt: session.modifiedAt });
      }),
    );
  }
  return hits
    .sort((a, b) => b.match.score - a.match.score || b.modifiedAt.localeCompare(a.modifiedAt))
    .slice(0, limit)
    .map((hit) => hit.match);
}
