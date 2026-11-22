/**
 * Pi's JSONL session files on this machine: metadata scan, listing per cwd or across cwds, lookup by id, archive and
 * restore (C7), deletion.
 *
 * - Consumes: session files under the roots from `paths.ts`, the M1 lock, sidecars from `state.ts`
 * - Provides: `sessionMeta`, `listSessions`, `findSession`, `moveSession`, `deleteSession`, `readEntries` behind
 *   `LocalHost`
 */
import { mkdir, readdir, realpath, rename, rm, stat } from "node:fs/promises";
import { dirname, join, relative, resolve } from "node:path";
import { FACTORY_STATES, type DeleteResult, type FactoryState, type MoveResult, type SessionEntries, type SessionMeta, type SessionUsage } from "../index.ts";
import { tryLockFile, type FileLock } from "./lock.ts";
import { archiveRoot, cwdSessionDir, factorySessionsRoot, isInside, sessionsRoot } from "./paths.ts";
import { moveSidecar, removeSidecar } from "./state.ts";

type Summary = Pick<
  SessionMeta,
  "sessionId" | "cwd" | "createdAt" | "firstMessage" | "messageCount" | "tokens" | "parentSession" | "factory" | "usage"
>;
type Header = Pick<Summary, "sessionId" | "cwd" | "createdAt" | "parentSession">;

const FIRST_MESSAGE_MAX_CHARS = 300;
const MESSAGE_LINE_PREFIX = '{"type":"message"';
/** Pi writes entries with `JSON.stringify`, so a `factory-state` line holds exactly these two substrings. */
const CUSTOM_LINE_PREFIX = '{"type":"custom",';
const FACTORY_STATE_MARK = '"customType":"factory-state"';
const FACTORY_REVIEW_MARK = '"customType":"factory-review"';
/** An assistant line's usage as Pi writes it (`JSON.stringify`, fixed key order): tokens, then `cost.total`. */
const USAGE_PATTERN = /"usage":\{"input":(\d+),"output":(\d+),"cacheRead":(\d+),"cacheWrite":(\d+),[^{}]*"cost":\{[^{}]*"total":([0-9.eE+-]+)/;
const summaryCache = new Map<string, { mtimeMs: number; size: number; summary: Summary }>();

/** Text of a user message's content (string or text blocks). */
function messageText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .filter((block) => block?.type === "text" && typeof block.text === "string")
    .map((block) => block.text as string)
    .join(" ");
}

/** First user text of a message line, empty when the line is not a user message with text. */
function userText(line: string): string {
  try {
    const entry = JSON.parse(line);
    if (entry?.message?.role !== "user") return "";
    return messageText(entry.message.content).trim();
  } catch {
    return "";
  }
}

/** The context size an assistant message line reports (`usage.totalTokens`), undefined when it has none. */
function lastTotalTokens(line: string): number | undefined {
  const found = [...line.matchAll(/"totalTokens":(\d+)/g)].pop();
  return found ? Number(found[1]) : undefined;
}

/** The usage of an assistant message line, undefined when it reports none. */
function answerUsage(line: string): SessionUsage | undefined {
  const found = USAGE_PATTERN.exec(line);
  if (!found) return undefined;
  const [, input, output, cacheRead, cacheWrite, cost] = found.map(Number);
  return { input: input + cacheRead + cacheWrite, output, cost };
}

/** The usage a `factory-review` line records for its reviewer child, undefined when the line is not one. */
function reviewUsage(line: string): SessionUsage | undefined {
  try {
    const usage = JSON.parse(line)?.data?.usage;
    if (typeof usage?.input !== "number" || typeof usage.output !== "number" || typeof usage.cost !== "number") return undefined;
    return { input: usage.input + (usage.cacheRead ?? 0) + (usage.cacheWrite ?? 0), output: usage.output, cost: usage.cost };
  } catch {
    return undefined;
  }
}

/** The state a `factory-state` line records, undefined when the line is not one or lacks a field. */
function factoryState(line: string): FactoryState | undefined {
  try {
    const entry = JSON.parse(line);
    const data = entry?.data;
    if (entry?.type !== "custom" || entry.customType !== "factory-state" || typeof entry.timestamp !== "string") return undefined;
    if (!FACTORY_STATES.includes(data?.state) || typeof data.round !== "number" || typeof data.reason !== "string") return undefined;
    return { state: data.state, round: data.round, reason: data.reason, at: entry.timestamp };
  } catch {
    return undefined;
  }
}

/** Header fields of a session file's first line, undefined when it is not a Pi session header. */
function parseHeader(line: string): Header | undefined {
  try {
    const header = JSON.parse(line);
    if (header?.type !== "session" || typeof header.id !== "string") return undefined;
    return {
      sessionId: header.id,
      cwd: typeof header.cwd === "string" ? header.cwd : "",
      createdAt: new Date(header.timestamp).toISOString(),
      ...(typeof header.parentSession === "string" ? { parentSession: header.parentSession } : {}),
    };
  } catch {
    return undefined;
  }
}

/**
 * Stream a session file once: parse the header, count message lines by prefix, parse only until the first user text
 * and the `factory-state` lines (matched by substring first, the last valid one in file order wins).
 */
async function scanSession(sessionPath: string): Promise<Summary | undefined> {
  const decoder = new TextDecoder();
  let header: Header | undefined;
  let factory: FactoryState | undefined;
  let firstMessage = "";
  let messageCount = 0;
  let tokens = 0;
  const usage: SessionUsage = { input: 0, output: 0, cost: 0 };
  const add = (more: SessionUsage | undefined) => {
    if (!more) return;
    usage.input += more.input;
    usage.output += more.output;
    usage.cost += more.cost;
  };
  let pending = "";
  const consume = (line: string): boolean => {
    if (!header) {
      header = parseHeader(line);
      return header !== undefined;
    }
    if (!line.startsWith(MESSAGE_LINE_PREFIX)) {
      if (line.startsWith(CUSTOM_LINE_PREFIX) && line.includes(FACTORY_STATE_MARK)) factory = factoryState(line) ?? factory;
      if (line.startsWith(CUSTOM_LINE_PREFIX) && line.includes(FACTORY_REVIEW_MARK)) add(reviewUsage(line));
      return true;
    }
    messageCount++;
    if (line.includes('"role":"assistant"')) {
      tokens = lastTotalTokens(line) ?? tokens;
      add(answerUsage(line));
    }
    if (!firstMessage && line.includes('"role":"user"')) firstMessage = userText(line);
    return true;
  };
  for await (const chunk of Bun.file(sessionPath).stream()) {
    const lines = (pending + decoder.decode(chunk, { stream: true })).split("\n");
    pending = lines.pop() ?? "";
    for (const line of lines) if (!consume(line)) return undefined;
  }
  pending += decoder.decode();
  if (pending && !consume(pending)) return undefined;
  if (!header) return undefined;
  const firstText = firstMessage || "(no messages)";
  return {
    ...header,
    firstMessage: firstText.length > FIRST_MESSAGE_MAX_CHARS ? `${firstText.slice(0, FIRST_MESSAGE_MAX_CHARS)}…` : firstText,
    messageCount,
    tokens,
    ...(factory ? { factory } : {}),
    ...(usage.input + usage.output > 0 ? { usage } : {}),
  };
}

/** Summary of a session file, cached by mtime and size. */
async function summarize(sessionPath: string): Promise<{ summary: Summary; mtime: Date } | undefined> {
  let stats;
  try {
    stats = await stat(sessionPath);
  } catch {
    return undefined;
  }
  if (!stats.isFile()) return undefined;
  const cached = summaryCache.get(sessionPath);
  if (cached && cached.mtimeMs === stats.mtimeMs && cached.size === stats.size) {
    return { summary: cached.summary, mtime: stats.mtime };
  }
  const summary = await scanSession(sessionPath).catch(() => undefined);
  if (!summary) return undefined;
  summaryCache.set(sessionPath, { mtimeMs: stats.mtimeMs, size: stats.size, summary });
  return { summary, mtime: stats.mtime };
}

/** Metadata of a session file, undefined when it is missing or not a Pi session. */
export async function sessionMeta(sessionPath: string): Promise<SessionMeta | undefined> {
  const absolutePath = resolve(sessionPath);
  const scanned = await summarize(absolutePath);
  if (!scanned) return undefined;
  return {
    ...scanned.summary,
    sessionPath: absolutePath,
    realPath: await realpath(absolutePath).catch(() => absolutePath),
    modifiedAt: scanned.mtime.toISOString(),
    archived: isInside(archiveRoot(), absolutePath),
  };
}

/** Newest first by modification, then creation. */
function newestFirst(a: SessionMeta, b: SessionMeta): number {
  return b.modifiedAt.localeCompare(a.modifiedAt) || b.createdAt.localeCompare(a.createdAt);
}

/**
 * Sessions of `cwd` (every cwd when null), newest first. Live ones come from the live root, plus the factory root with
 * `factory`. Archived ones come from the archive root, which never holds a factory-root session. The factory root is
 * grouped by repository, not cwd: it is scanned whole and filtered by the header's cwd.
 */
export async function listSessions(cwd: string | null, archived: boolean, factory = false): Promise<SessionMeta[]> {
  const withFactory = factory && !archived;
  if (cwd === null) {
    const roots = [archived ? archiveRoot() : sessionsRoot(), ...(withFactory ? [factorySessionsRoot()] : [])];
    const files = (await Promise.all(roots.map((root) => sessionFiles(root, ".jsonl")))).flat();
    const metas = await Promise.all(files.map((path) => sessionMeta(path)));
    return metas.filter((meta): meta is SessionMeta => meta !== undefined).sort(newestFirst);
  }
  const paths = [
    ...(await cwdSessionFiles(cwdSessionDir(archived ? archiveRoot() : sessionsRoot(), cwd))),
    ...(withFactory ? await sessionFiles(factorySessionsRoot(), ".jsonl") : []),
  ];
  const metas = await Promise.all(paths.map((path) => sessionMeta(path)));
  const wantedCwd = resolve(cwd);
  return metas
    .filter((meta): meta is SessionMeta => meta !== undefined && resolve(meta.cwd || "/") === wantedCwd)
    .sort(newestFirst);
}

/** Session files directly in a cwd's session directory, none when it is missing. */
async function cwdSessionFiles(dir: string): Promise<string[]> {
  let names: string[];
  try {
    names = await readdir(dir);
  } catch {
    return [];
  }
  return names.filter((name) => name.endsWith(".jsonl")).map((name) => join(dir, name));
}

/** Files ending in `suffix` directly under `dir`, and its subdirectories. */
async function matchingFiles(dir: string, suffix: string): Promise<{ files: string[]; subdirs: string[] }> {
  try {
    const entries = await readdir(dir, { withFileTypes: true });
    return {
      files: entries.filter((entry) => entry.isFile() && entry.name.endsWith(suffix)).map((entry) => join(dir, entry.name)),
      subdirs: entries.filter((entry) => entry.isDirectory()).map((entry) => join(dir, entry.name)),
    };
  } catch {
    return { files: [], subdirs: [] };
  }
}

/** Files ending in `suffix` under `root`: directly in it (flat session dir) and one level down (per-cwd dirs). */
async function sessionFiles(root: string, suffix: string): Promise<string[]> {
  const top = await matchingFiles(root, suffix);
  const nested = await Promise.all(top.subdirs.map((dir) => matchingFiles(dir, suffix)));
  return [...top.files, ...nested.flatMap((level) => level.files)];
}

/**
 * Path of the session file with Pi session id `sessionId` (`<timestamp>_<id>.jsonl`): live root first, then the factory
 * and archive roots.
 */
export async function findSession(sessionId: string): Promise<string | undefined> {
  if (!sessionId || /[/\\]/.test(sessionId) || sessionId.includes("..")) return undefined;
  for (const root of [sessionsRoot(), factorySessionsRoot(), archiveRoot()]) {
    for (const candidate of await sessionFiles(root, `_${sessionId}.jsonl`)) {
      const scanned = await summarize(candidate);
      if (scanned?.summary.sessionId === sessionId) return candidate;
    }
  }
  return undefined;
}

/**
 * Move a session file and its sidecar between the live and archive roots. The move holds the session's flock, the
 * one taken before a child starts, so a spawn and an archive exclude each other. A factory-root session is never
 * archived (`factory`).
 */
export async function moveSession(sessionPath: string, toArchive: boolean): Promise<MoveResult> {
  if (isInside(factorySessionsRoot(), sessionPath)) return { ok: false, error: "factory" };
  const fromRoot = toArchive ? sessionsRoot() : archiveRoot();
  const toRoot = toArchive ? archiveRoot() : sessionsRoot();
  if (!isInside(fromRoot, sessionPath)) return { ok: false, error: "not_in_root" };
  let lock: FileLock | undefined;
  try {
    lock = tryLockFile(sessionPath);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return { ok: false, error: "not_found" };
    throw error;
  }
  if (!lock) return { ok: false, error: "locked" };
  try {
    const target = join(toRoot, relative(fromRoot, sessionPath));
    if (await Bun.file(target).exists()) return { ok: false, error: "exists" };
    await mkdir(dirname(target), { recursive: true });
    await rename(sessionPath, target);
    try {
      await moveSidecar(sessionPath, target);
    } catch (error) {
      await rename(target, sessionPath);
      throw error;
    }
    summaryCache.delete(sessionPath);
    return { ok: true, path: target };
  } finally {
    lock.release();
  }
}

/**
 * Remove a session file and its sidecar. Like a move it holds the session's flock, so a spawn and a delete exclude
 * each other.
 */
export async function deleteSession(sessionPath: string): Promise<DeleteResult> {
  let lock: FileLock | undefined;
  try {
    lock = tryLockFile(sessionPath);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return { ok: false, error: "not_found" };
    throw error;
  }
  if (!lock) return { ok: false, error: "locked" };
  try {
    await rm(sessionPath);
    await removeSidecar(sessionPath);
    summaryCache.delete(sessionPath);
    return { ok: true };
  } finally {
    lock.release();
  }
}

/** Entries of a session file in append order (the header line excluded), the last one the leaf. */
export async function readEntries(sessionPath: string): Promise<SessionEntries> {
  let text: string;
  try {
    text = await Bun.file(sessionPath).text();
  } catch {
    return { entries: [], leafId: null };
  }
  const entries: Record<string, unknown>[] = [];
  for (const line of text.split("\n")) {
    if (!line) continue;
    try {
      const entry = JSON.parse(line);
      if (entry && typeof entry === "object" && entry.type !== "session" && typeof entry.id === "string") entries.push(entry);
    } catch {
      // A half-written last line
    }
  }
  return { entries, leafId: (entries.at(-1)?.id as string | undefined) ?? null };
}
