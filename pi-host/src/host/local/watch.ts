/**
 * Sessions-directory watcher (M25): reports session files written by anyone, the Pi TUI included, so lists update
 * live, and the factory plugin's session, task record and runner lock files, so the factory board follows tasks a
 * CLI creates, advances or removes, and sees a runner end (it removes its lock on exit). It only reports changed files; it is not a code-reload watcher (M4).
 *
 * - Consumes: a recursive `fs.watch` on the live session root and the factory's data root from `paths.ts`
 * - Provides: `watchSessions(onChange)` behind `LocalHost.watchSessions`
 *
 * Changes are batched: the first change opens a 500 ms window and its end reports every session file changed in
 * it, so a burst of writes is one report and a streaming session reports at most twice a second. A change the
 * platform reports without a file name reports an empty list (unknown files). Without a root yet, or after a watch
 * error, the watch of that root is retried every 5 s.
 *
 * Ported from omp-web `lib/session-watcher.ts` (MIT, Copyright (c) 2026 agegr, https://github.com/ompweb-dev/ompweb).
 */
import { watch, type FSWatcher } from "node:fs";
import { join, sep } from "node:path";
import { factoryRoot, sessionsRoot } from "./paths.ts";

export const DEBOUNCE_MS = 500;
const RETRY_MS = 5_000;

/**
 * A changed file worth a report: a session file anywhere, or one of the factory's task records (`tasks/<id>.json`)
 * or runner locks (`tasks/<id>.lock`).
 */
function isReported(root: string, filename: string): boolean {
  if (filename.endsWith(".jsonl")) return true;
  return root === factoryRoot() && filename.startsWith(`tasks${sep}`) && (filename.endsWith(".json") || filename.endsWith(".lock"));
}

/** Watch the live session root and the factory's data root until the returned function is called. */
export function watchSessions(onChange: (sessionPaths: string[]) => void): () => void {
  let flushTimer: Timer | undefined;
  let pending = new Set<string>();
  let unknown = false;

  const flush = (): void => {
    flushTimer = undefined;
    const paths = unknown ? [] : [...pending];
    pending = new Set();
    unknown = false;
    onChange(paths);
  };

  const record = (root: string, filename: string | null | undefined): void => {
    // Bun reports an event without a name as null or undefined: either means "something changed, rescan".
    if (filename == null) unknown = true;
    else if (isReported(root, filename)) pending.add(join(root, filename));
    else return;
    flushTimer ??= setTimeout(flush, DEBOUNCE_MS);
  };

  const stops = [sessionsRoot(), factoryRoot()].map((root) => watchRoot(root, record));
  return () => {
    clearTimeout(flushTimer);
    for (const stop of stops) stop();
  };
}

/** Recursive watch of `root`, retried every 5 s while it is missing or after an error. */
function watchRoot(root: string, record: (root: string, filename: string | null | undefined) => void): () => void {
  let watcher: FSWatcher | undefined;
  let retryTimer: Timer | undefined;
  let stopped = false;

  const start = (): void => {
    retryTimer = undefined;
    if (stopped) return;
    try {
      watcher = watch(root, { recursive: true, persistent: false }, (_event, filename) => record(root, filename));
      watcher.on("error", () => {
        watcher?.close();
        watcher = undefined;
        retry();
      });
    } catch {
      // No session root yet.
      retry();
    }
  };

  const retry = (): void => {
    if (stopped || retryTimer) return;
    retryTimer = setTimeout(start, RETRY_MS);
    retryTimer.unref();
  };

  start();
  return () => {
    stopped = true;
    clearTimeout(retryTimer);
    watcher?.close();
  };
}
