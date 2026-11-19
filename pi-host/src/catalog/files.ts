/**
 * `@` file completion for a cwd (M7): candidate files plus fuzzy ranking.
 *
 * - Consumes: `git ls-files --cached --others --exclude-standard` in the cwd, a bounded directory walk outside git
 * - Provides: `listFiles(cwd)` (relative paths, briefly cached) and `rankFiles(files, query, limit)`
 */
import { readdir } from "node:fs/promises";
import { join } from "node:path";

export type RankedFile = { path: string; score: number };

const CACHE_TTL_MS = 5_000;
const WALK_MAX_FILES = 20_000;
const WALK_MAX_DEPTH = 12;
const WALK_SKIPPED_DIRS: Record<string, true> = { node_modules: true, __pycache__: true };
const MAX_QUERY_CHARS = 64;
const MAX_PATH_CHARS = 512;
const SEPARATORS = "/\\_-. ";

const CHAR_SCORE = 10;
const BOUNDARY_BONUS = 8;
const CONSECUTIVE_BONUS = 12;
const GAP_PENALTY = 1;
const LEADING_PENALTY = 0.5;
const BASENAME_BONUS = 40;
const UNMATCHED_BASENAME_PENALTY = 0.2;
const NO_MATCH = Number.NEGATIVE_INFINITY;

const fileCache = new Map<string, { at: number; files: Promise<string[]> }>();

/** Tracked and untracked-not-ignored files, undefined when `cwd` is not in a git work tree. */
async function gitFiles(cwd: string): Promise<string[] | undefined> {
  const child = Bun.spawn(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], {
    cwd,
    stdout: "pipe",
    stderr: "ignore",
    stdin: "ignore",
  });
  const output = await new Response(child.stdout).text();
  if ((await child.exited) !== 0) return undefined;
  return [...new Set(output.split("\0").filter(Boolean))];
}

/** Breadth-first walk skipping hidden and dependency dirs, capped in files and depth. */
async function walkFiles(cwd: string): Promise<string[]> {
  const files: string[] = [];
  let level: string[] = [""];
  for (let depth = 0; depth <= WALK_MAX_DEPTH && level.length > 0; depth++) {
    const next: string[] = [];
    for (const dir of level) {
      let entries;
      try {
        entries = await readdir(join(cwd, dir), { withFileTypes: true });
      } catch {
        continue;
      }
      for (const entry of entries) {
        const path = dir ? `${dir}/${entry.name}` : entry.name;
        if (entry.isDirectory()) {
          if (!entry.name.startsWith(".") && !Object.hasOwn(WALK_SKIPPED_DIRS, entry.name)) next.push(path);
        } else if (entry.isFile()) {
          files.push(path);
          if (files.length >= WALK_MAX_FILES) return files;
        }
      }
    }
    level = next;
  }
  return files;
}

/** Candidate files of `cwd`, relative paths, cached for a few seconds while the user types. */
export function listFiles(cwd: string): Promise<string[]> {
  const cached = fileCache.get(cwd);
  if (cached && Date.now() - cached.at < CACHE_TTL_MS) return cached.files;
  const files = gitFiles(cwd).then((tracked) => tracked ?? walkFiles(cwd));
  fileCache.set(cwd, { at: Date.now(), files });
  files.catch(() => fileCache.delete(cwd));
  return files;
}

/** True when position `index` of `text` starts a word (start, after a separator, or a camelCase hump). */
function isBoundary(text: string, index: number): boolean {
  if (index === 0) return true;
  const previous = text[index - 1]!;
  if (SEPARATORS.includes(previous)) return true;
  const current = text[index]!;
  return previous === previous.toLowerCase() && current !== current.toLowerCase();
}

/** Best subsequence alignment score of `query` (lowercase) in `text`, -Infinity without a match. */
export function fuzzyScore(query: string, text: string): number {
  const lower = text.toLowerCase();
  let probe = 0;
  for (let index = 0; index < lower.length && probe < query.length; index++) {
    if (lower[index] === query[probe]) probe++;
  }
  if (probe < query.length) return NO_MATCH;
  let previousRow = new Array<number>(lower.length).fill(NO_MATCH);
  for (let row = 0; row < query.length; row++) {
    const currentRow = new Array<number>(lower.length).fill(NO_MATCH);
    let bestBeforeGap = NO_MATCH;
    for (let column = 0; column < lower.length; column++) {
      if (row > 0 && column >= 2) {
        bestBeforeGap = Math.max(bestBeforeGap, previousRow[column - 2]! + GAP_PENALTY * (column - 2));
      }
      if (lower[column] !== query[row]) continue;
      const charScore = CHAR_SCORE + (isBoundary(text, column) ? BOUNDARY_BONUS : 0);
      if (row === 0) {
        currentRow[column] = charScore - LEADING_PENALTY * column;
        continue;
      }
      let beforeSeparators = column - 1;
      while (beforeSeparators >= 0 && SEPARATORS.includes(lower[beforeSeparators]!)) beforeSeparators--;
      const consecutive = beforeSeparators >= 0 ? previousRow[beforeSeparators]! + CONSECUTIVE_BONUS : NO_MATCH;
      const gapped = bestBeforeGap - GAP_PENALTY * (column - 1);
      currentRow[column] = charScore + Math.max(consecutive, gapped);
    }
    previousRow = currentRow;
  }
  return Math.max(...previousRow);
}

/**
 * Score of one path: whole-path alignment, or a basename alignment plus a bonus minus a little
 * per unmatched basename character, whichever is higher.
 */
function pathScore(query: string, path: string): number {
  const clipped = path.length > MAX_PATH_CHARS ? path.slice(-MAX_PATH_CHARS) : path;
  const basename = clipped.slice(clipped.lastIndexOf("/") + 1);
  const basenameScore =
    fuzzyScore(query, basename) + BASENAME_BONUS - UNMATCHED_BASENAME_PENALTY * (basename.length - query.length);
  return Math.max(fuzzyScore(query, clipped), basenameScore);
}

/** Paths ranked against `query` (subsequence, basename and contiguous hits first), at most `limit`. */
export function rankFiles(files: string[], query: string, limit: number): RankedFile[] {
  const needle = query.replace(/\s+/g, "").toLowerCase().slice(0, MAX_QUERY_CHARS);
  if (!needle) {
    return [...files]
      .sort((a, b) => a.length - b.length || a.localeCompare(b))
      .slice(0, limit)
      .map((path) => ({ path, score: 0 }));
  }
  const ranked: RankedFile[] = [];
  for (const path of files) {
    const score = pathScore(needle, path);
    if (score !== NO_MATCH) ranked.push({ path, score: Math.round(score * 10) / 10 });
  }
  return ranked
    .sort((a, b) => b.score - a.score || a.path.length - b.path.length || a.path.localeCompare(b.path))
    .slice(0, limit);
}
