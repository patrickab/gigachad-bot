/**
 * Session catalog over Pi's JSONL session files: listing, lookup by id, archive and restore (M9, C7).
 *
 * - Consumes: session files under the roots from `paths.ts`, `LiveSessions` for live status, sidecars from `state.ts`
 * - Provides: `SessionInfo` (contract shape), `sessionInfo`, `listSessions`, `findSession`, `moveSession`
 */
import { mkdir, readdir, rename, stat } from "node:fs/promises";
import { dirname, join, relative, resolve } from "node:path";
import type { LiveSessions } from "../supervisor/index.ts";
import { archiveRoot, cwdSessionDir, isInside, sessionsRoot } from "./paths.ts";
import { moveSidecar, readState } from "./state.ts";

export type LiveStatus = { running: boolean; waiting: boolean; attached: number };

export type SessionInfo = {
  sessionId: string;
  sessionPath: string;
  cwd: string;
  machine: "local";
  createdAt: string;
  modifiedAt: string;
  firstMessage: string;
  messageCount: number;
  parentSession?: string;
  live: LiveStatus | null;
  pinned: boolean;
  archived: boolean;
};

type Summary = Pick<SessionInfo, "sessionId" | "cwd" | "createdAt" | "firstMessage" | "messageCount" | "parentSession">;

const FIRST_MESSAGE_MAX_CHARS = 300;
const MESSAGE_LINE_PREFIX = '{"type":"message"';
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

/** Header fields of a session file's first line, undefined when it is not a Pi session header. */
function parseHeader(line: string): Omit<Summary, "firstMessage" | "messageCount"> | undefined {
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

/** Stream a session file once: parse the header, count message lines by prefix, parse only until the first user text. */
async function scanSession(sessionPath: string): Promise<Summary | undefined> {
  const decoder = new TextDecoder();
  let header: Omit<Summary, "firstMessage" | "messageCount"> | undefined;
  let firstMessage = "";
  let messageCount = 0;
  let pending = "";
  const consume = (line: string): boolean => {
    if (!header) {
      header = parseHeader(line);
      return header !== undefined;
    }
    if (!line.startsWith(MESSAGE_LINE_PREFIX)) return true;
    messageCount++;
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

/** Contract `SessionInfo` for a session file, undefined when it is missing or not a Pi session. */
export async function sessionInfo(sessionPath: string, live: LiveSessions): Promise<SessionInfo | undefined> {
  const absolutePath = resolve(sessionPath);
  const scanned = await summarize(absolutePath);
  if (!scanned) return undefined;
  const { pinned } = await readState(absolutePath);
  return {
    ...scanned.summary,
    sessionPath: absolutePath,
    machine: "local",
    modifiedAt: scanned.mtime.toISOString(),
    live: live.liveStatus(absolutePath) ?? null,
    pinned,
    archived: isInside(archiveRoot(), absolutePath),
  };
}

/** Sessions of `cwd`, newest first. */
export async function listSessions(cwd: string, archived: boolean, live: LiveSessions): Promise<SessionInfo[]> {
  const dir = cwdSessionDir(archived ? archiveRoot() : sessionsRoot(), cwd);
  let names: string[];
  try {
    names = await readdir(dir);
  } catch {
    return [];
  }
  const infos = await Promise.all(
    names.filter((name) => name.endsWith(".jsonl")).map((name) => sessionInfo(join(dir, name), live)),
  );
  const wantedCwd = resolve(cwd);
  return infos
    .filter((info): info is SessionInfo => info !== undefined && resolve(info.cwd || "/") === wantedCwd)
    .sort((a, b) => b.modifiedAt.localeCompare(a.modifiedAt) || b.createdAt.localeCompare(a.createdAt));
}

/** Session files named for `sessionId` directly under `dir`. */
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

/** Path of the session file with Pi session id `sessionId` (`<timestamp>_<id>.jsonl`), live root first. */
export async function findSession(sessionId: string): Promise<string | undefined> {
  if (!sessionId || /[/\\]/.test(sessionId) || sessionId.includes("..")) return undefined;
  const suffix = `_${sessionId}.jsonl`;
  for (const root of [sessionsRoot(), archiveRoot()]) {
    const top = await matchingFiles(root, suffix);
    const nested = await Promise.all(top.subdirs.map((dir) => matchingFiles(dir, suffix)));
    for (const candidate of [...top.files, ...nested.flatMap((level) => level.files)]) {
      const scanned = await summarize(candidate);
      if (scanned?.summary.sessionId === sessionId) return candidate;
    }
  }
  return undefined;
}

export type MoveResult = { ok: true; path: string } | { ok: false; error: "live" | "exists" | "not_in_root" };

/** Move a session file and its sidecar between the live and archive roots, refused while live (C7). */
export async function moveSession(sessionPath: string, toArchive: boolean, live: LiveSessions): Promise<MoveResult> {
  const fromRoot = toArchive ? sessionsRoot() : archiveRoot();
  const toRoot = toArchive ? archiveRoot() : sessionsRoot();
  if (!isInside(fromRoot, sessionPath)) return { ok: false, error: "not_in_root" };
  if (live.liveStatus(sessionPath)) return { ok: false, error: "live" };
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
}
