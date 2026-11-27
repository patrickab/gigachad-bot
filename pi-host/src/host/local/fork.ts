/**
 * Fork and clone of a session file the way Pi 1.5.2 writes them (`SessionManager.createBranchedSession`, D11).
 *
 * - Consumes: a source session file, the live root from `paths.ts`
 * - Provides: `forkSession` behind `LocalHost`: a new file with the branch from the root to the fork entry
 *
 * Pi's rules: label entries on the branch are dropped and the labels of kept entries are written again after the
 * branch, the branch is relinked so labels in between leave no gaps, a compaction whose first kept entry was a label
 * points to the entry that followed it. The source header's version is kept: Pi migrates older files on load.
 */
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { pathToRoot, type Entry } from "../entries.ts";
import type { ForkRequest, ForkResult } from "../index.ts";
import { cwdSessionDir, sessionsRoot } from "./paths.ts";

function isEntry(value: unknown): value is Entry {
  const entry = value as Entry | null;
  return typeof entry?.type === "string" && typeof entry.id === "string" && entry.type !== "session";
}

/** Unparsable lines are skipped, as Pi does. */
function parseLines(text: string): unknown[] {
  const records: unknown[] = [];
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    try {
      records.push(JSON.parse(line));
    } catch {
      // a partial last line of a file Pi is appending to
    }
  }
  return records;
}

/** Pi's `generateId`: 8 hex characters not taken yet. */
function newEntryId(taken: Set<string>): string {
  for (let attempt = 0; attempt < 100; attempt++) {
    const id = crypto.randomUUID().slice(0, 8);
    if (!taken.has(id)) return id;
  }
  return crypto.randomUUID();
}

/** The fork's entries: the branch from the root to `entryId` without labels, then the kept entries' labels. */
function branchEntries(entries: Entry[], entryId: string): Entry[] | undefined {
  const byId = new Map(entries.map((entry) => [entry.id, entry]));
  const labels = new Map<string, { label: unknown; timestamp: unknown }>();
  for (const entry of entries) {
    if (entry.type !== "label" || typeof entry.targetId !== "string") continue;
    if (entry.label) labels.set(entry.targetId, { label: entry.label, timestamp: entry.timestamp });
    else labels.delete(entry.targetId);
  }

  const path = pathToRoot(byId, entryId);
  if (path.length === 0) return undefined;

  const kept: Entry[] = [];
  const replacementByLabelId = new Map<string, string>();
  const pendingLabelIds: string[] = [];
  let parentId: string | null = null;
  for (const entry of path) {
    if (entry.type === "label") {
      pendingLabelIds.push(entry.id);
      continue;
    }
    for (const labelId of pendingLabelIds) replacementByLabelId.set(labelId, entry.id);
    pendingLabelIds.length = 0;
    const firstKept = entry.firstKeptEntryId as string;
    kept.push(
      entry.type === "compaction" && firstKept !== entry.id
        ? { ...entry, parentId, firstKeptEntryId: replacementByLabelId.get(firstKept) ?? firstKept }
        : { ...entry, parentId },
    );
    parentId = entry.id;
  }

  const taken = new Set(kept.map((entry) => entry.id));
  const labelEntries: Entry[] = [];
  for (const [targetId, { label, timestamp }] of labels) {
    if (!taken.has(targetId)) continue;
    const id = newEntryId(taken);
    taken.add(id);
    labelEntries.push({ type: "label", id, parentId, timestamp, targetId, label });
    parentId = id;
  }
  return [...kept, ...labelEntries];
}

export async function forkSession({ sessionPath, entryId }: ForkRequest): Promise<ForkResult> {
  const sourcePath = resolve(sessionPath);
  let text: string;
  try {
    text = await readFile(sourcePath, "utf8");
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code === "ENOENT" || code === "EISDIR") return { ok: false, error: "not_found" };
    throw error;
  }
  const [header, ...records] = parseLines(text) as [Record<string, unknown> | undefined, ...unknown[]];
  if (header?.type !== "session" || typeof header.id !== "string" || typeof header.cwd !== "string") {
    return { ok: false, error: "not_found" };
  }
  const entries = branchEntries(records.filter(isEntry), entryId);
  if (!entries) return { ok: false, error: "no_entry" };

  const id = Bun.randomUUIDv7();
  const timestamp = new Date().toISOString();
  const forkHeader = { type: "session", version: header.version ?? 1, id, timestamp, cwd: header.cwd, parentSession: sourcePath };
  const dir = cwdSessionDir(sessionsRoot(), header.cwd);
  const path = join(dir, `${timestamp.replace(/[:.]/g, "-")}_${id}.jsonl`);
  await mkdir(dir, { recursive: true });
  await writeFile(path, [forkHeader, ...entries].map((record) => `${JSON.stringify(record)}\n`).join(""), { flag: "wx" });
  return { ok: true, path };
}
