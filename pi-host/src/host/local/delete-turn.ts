/**
 * Deleting a turn from a session file on this machine (conversation unification, wave pi/1): delete means gone from
 * disk, so nothing of the turn may stay in the file, the caches derived from it, the media store or `.pi/pastes`.
 *
 * - Consumes: the session file, `removeTurn` (`turns.ts`), the M1 lock, the other session files of the same cwd (live
 *   and archive roots, substring tests only)
 * - Provides: `deleteTurn` behind `LocalHost.deleteTurn`
 *
 * The new file is written beside the old one under a name not ending in `.jsonl` (the session watcher and listings
 * skip it), synced, then renamed over it: a crash leaves the old or the new file, no copy of the old one is kept.
 * Entries `removeTurn` passes through unchanged keep their original line bytes, lines that are no entry stay as they are.
 * Once the file is renamed the caches are always forgotten and a cleanup that fails is named in `notRemoved`.
 */
import { createHash } from "node:crypto";
import { lstat, open, readdir, readFile, realpath, rename, rm, stat } from "node:fs/promises";
import { basename, dirname, isAbsolute, join } from "node:path";
import type { DeleteTurnRequest, DeleteTurnResult } from "../index.ts";
import { tryLockFile, type FileLock } from "./lock.ts";
import { archiveRoot, cwdSessionDir, sessionsRoot } from "./paths.ts";
import { forgetTranscript } from "./search.ts";
import { forgetSummary } from "./sessions.ts";
import { removeTurn, type TurnEntry } from "./turns.ts";

/** A `@` mention of a pasted video (`lib/pi_pastes.py`, `paste_mention`), relative to the session's cwd. */
const PASTE_MENTION = /\.pi\/pastes\/([\w.-]+)/g;

/**
 * The file's lines (without the final newline) and where its entries sit: `slots[i]` is the line of `entries[i]`.
 * Lines that do not parse, entries without an id and blank lines are no entries: a rewrite keeps them byte for byte.
 */
type Parsed = { headerLine: string; cwd: string; lines: string[]; entries: TurnEntry[]; slots: number[]; raw: Map<TurnEntry, string> };

/** Header line, cwd, lines and entries of a session file's text; undefined without a session header. */
function parseSession(text: string): Parsed | undefined {
  const lines = text.split("\n");
  if (lines.at(-1) === "") lines.pop();
  const headerIndex = lines.findIndex(Boolean);
  const headerLine = lines[headerIndex] ?? "";
  let header: unknown;
  try {
    header = JSON.parse(headerLine);
  } catch {
    return undefined;
  }
  const { type, cwd } = (header ?? {}) as { type?: unknown; cwd?: unknown };
  if (type !== "session") return undefined;
  const entries: TurnEntry[] = [];
  const slots: number[] = [];
  const raw = new Map<TurnEntry, string>();
  for (let index = headerIndex + 1; index < lines.length; index++) {
    const line = lines[index]!;
    let entry: TurnEntry;
    try {
      entry = JSON.parse(line);
    } catch {
      continue; // A half-written line, Pi skips it too
    }
    if (typeof entry?.id !== "string" || entry.type === "session") continue;
    entries.push(entry);
    slots.push(index);
    raw.set(entry, line);
  }
  return { headerLine, cwd: typeof cwd === "string" ? cwd : "", lines, entries, slots, raw };
}

/**
 * The file's text with `kept` in the lines of the entries: the removed entries' lines go, every other line stays where
 * it was. `kept` is in file order but for a moved leaf, so it fills the surviving entry lines in order.
 */
function rewritten(parsed: Parsed, kept: TurnEntry[]): string {
  const removed = new Set(parsed.entries.filter((entry) => !kept.some((next) => next.id === entry.id)));
  const surviving = parsed.slots.filter((_, index) => !removed.has(parsed.entries[index]!));
  const lines: (string | undefined)[] = [...parsed.lines];
  for (const [index, entry] of parsed.entries.entries()) if (removed.has(entry)) lines[parsed.slots[index]!] = undefined;
  for (const [index, entry] of kept.entries()) lines[surviving[index]!] = parsed.raw.get(entry) ?? JSON.stringify(entry);
  return lines.filter((line) => line !== undefined).join("\n") + "\n";
}

/** Text of a message's content (string or text blocks). */
function textOf(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content.map((block) => (block?.type === "text" && typeof block.text === "string" ? block.text : "")).join("\n");
}

/** Base64 of the image blocks of tool results, the ones the media store holds (`media.ts`, `withMediaHashes`). */
function toolImages(entries: TurnEntry[]): string[] {
  const images: string[] = [];
  for (const entry of entries) {
    const message = entry.message as { role?: unknown; content?: unknown } | undefined;
    if (message?.role !== "toolResult" || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (block?.type === "image" && typeof block.data === "string" && block.data !== "") images.push(block.data);
    }
  }
  return images;
}

/** Relative paths of the pastes the user prompts among `entries` mention. */
function pasteMentions(entries: TurnEntry[]): string[] {
  const paths = new Set<string>();
  for (const entry of entries) {
    const message = entry.message as { role?: unknown; content?: unknown } | undefined;
    if (entry.type !== "message" || message?.role !== "user") continue;
    for (const [mention, name] of textOf(message.content).matchAll(PASTE_MENTION)) {
      if (name !== "." && name !== ".." && name !== ".gitignore") paths.add(mention);
    }
  }
  return [...paths];
}

/** Texts of the other session files of `cwd`, live and archived, read one at a time by `holds`. */
async function otherSessionFiles(cwd: string, sessionPath: string): Promise<string[]> {
  const files: string[] = [];
  for (const root of [sessionsRoot(), archiveRoot()]) {
    const dir = cwdSessionDir(root, cwd);
    const names = await readdir(dir).catch(() => [] as string[]);
    for (const name of names) {
      const path = join(dir, name);
      if (name.endsWith(".jsonl") && path !== sessionPath) files.push(path);
    }
  }
  return files;
}

/** The `needles` no other session file of the cwd contains, by plain substring test (no parse). */
async function unheld(needles: string[], files: string[]): Promise<string[]> {
  let left = needles;
  for (const file of files) {
    if (left.length === 0) break;
    const text = await readFile(file, "utf8").catch(() => "");
    left = left.filter((needle) => !text.includes(needle));
  }
  return left;
}

/** Write `text` to a new file `path` with `mode` and sync it, so the rename publishes complete bytes. */
async function writeSynced(path: string, text: string, mode: number): Promise<void> {
  const handle = await open(path, "wx", mode);
  try {
    await handle.writeFile(text);
    await handle.sync();
  } finally {
    await handle.close();
  }
}

/** Sync a folder after a rename in it, best effort (not every file system allows it). */
async function syncDir(dir: string): Promise<void> {
  const handle = await open(dir, "r").catch(() => undefined);
  await handle?.sync().catch(() => undefined);
  await handle?.close();
}

/**
 * `Host.deleteTurn` on this machine, see `DeleteTurnRequest` and `DeleteTurnResult`. `beforeRename` is a test hook
 * between the temp file's write and the rename.
 */
export async function deleteTurn(
  { sessionPath, entryId, through, dryRun }: DeleteTurnRequest,
  beforeRename: () => Promise<void> = async () => undefined,
): Promise<DeleteTurnResult> {
  // A symlinked session path is rewritten at its target, the link stays a link
  const filePath = await realpath(sessionPath).catch(() => undefined);
  if (!filePath) return { ok: false, error: "not_found" };
  let lock: FileLock | undefined;
  if (!dryRun) {
    try {
      lock = tryLockFile(filePath);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return { ok: false, error: "not_found" };
      throw error;
    }
    if (!lock) return { ok: false, error: "locked" };
  }
  try {
    const before = await stat(filePath);
    const parsed = parseSession(await readFile(filePath, "utf8"));
    if (!parsed) return { ok: false, error: "not_found" };
    const result = removeTurn(parsed.headerLine, parsed.entries, entryId, through);
    if (!result.ok) return { ok: false, error: result.reason };
    if (dryRun) return { ok: true, removedEntries: result.removed.length, removedMedia: [] };

    const text = rewritten(parsed, result.entries);
    const temp = join(dirname(filePath), `.${basename(filePath)}.${crypto.randomUUID()}.tmp`);
    try {
      await writeSynced(temp, text, before.mode & 0o777);
      await beforeRename();
      const now = await stat(filePath);
      if (now.ino !== before.ino || now.size !== before.size || now.mtimeMs !== before.mtimeMs) {
        await rm(temp, { force: true });
        return { ok: false, error: "changed" };
      }
      await rename(temp, filePath);
    } catch (error) {
      await rm(temp, { force: true });
      throw error;
    }
    // From here the file is rewritten: nothing below may skip forgetting the caches, each cleanup fails on its own
    await syncDir(dirname(filePath)).catch(() => undefined);
    for (const path of new Set([sessionPath, filePath])) {
      forgetSummary(path);
      forgetTranscript(path);
    }

    // What the range held and nothing else holds goes: in this file first, then in the cwd's other session files
    const notRemoved: string[] = [];
    const others = parsed.cwd ? otherSessionFiles(parsed.cwd, filePath) : Promise.resolve([]);
    const images = [...new Set(toolImages(result.removed))].filter((data) => !text.includes(data));
    const hashOf = (data: string) => createHash("sha256").update(Buffer.from(data, "base64")).digest("hex");
    let removedMedia: string[] = [];
    try {
      removedMedia = [...new Set((await unheld(images, await others)).map(hashOf))];
    } catch {
      notRemoved.push(...new Set(images.map(hashOf)));
    }
    if (isAbsolute(parsed.cwd)) {
      const pastes = pasteMentions(result.removed).filter((mention) => !text.includes(mention));
      let unused: string[] = [];
      try {
        unused = await unheld(pastes, await others);
      } catch {
        notRemoved.push(...pastes);
      }
      for (const mention of unused) {
        const path = join(parsed.cwd, mention);
        try {
          if ((await lstat(path).catch(() => undefined))?.isFile()) await rm(path, { force: true });
        } catch {
          notRemoved.push(mention);
        }
      }
    }
    return { ok: true, removedEntries: result.removed.length, removedMedia, ...(notRemoved.length > 0 && { notRemoved }) };
  } finally {
    lock?.release();
  }
}
