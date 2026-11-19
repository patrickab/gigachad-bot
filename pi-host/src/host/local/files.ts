/**
 * `@` completion candidates of a cwd on this machine (M7).
 *
 * - Consumes: `git ls-files --cached --others --exclude-standard` in the cwd, a bounded directory walk outside git
 * - Provides: `listFiles(cwd)`, relative paths, briefly cached while the user types
 */
import { readdir } from "node:fs/promises";
import { join } from "node:path";

const CACHE_TTL_MS = 5_000;
const WALK_MAX_FILES = 20_000;
const WALK_MAX_DEPTH = 12;
const WALK_SKIPPED_DIRS: Record<string, true> = { node_modules: true, __pycache__: true };

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

/** Candidate files of `cwd`, relative paths, cached for a few seconds. */
export function listFiles(cwd: string): Promise<string[]> {
  const cached = fileCache.get(cwd);
  if (cached && Date.now() - cached.at < CACHE_TTL_MS) return cached.files;
  const files = gitFiles(cwd).then((tracked) => tracked ?? walkFiles(cwd));
  fileCache.set(cwd, { at: Date.now(), files });
  files.catch(() => fileCache.delete(cwd));
  return files;
}
