/**
 * Sessions-directory watcher (M25): reports session files written by anyone, the Pi TUI included, so lists update
 * live. It only reports changed files; it is not a code-reload watcher (M4).
 *
 * - Consumes: a recursive `fs.watch` on the live session root from `paths.ts`
 * - Provides: `watchSessions(onChange)` behind `LocalHost.watchSessions`
 *
 * Changes are batched: the first change opens a 500 ms window and its end reports every session file changed in
 * it, so a burst of writes is one report and a streaming session reports at most twice a second. A change the
 * platform reports without a file name reports an empty list (unknown files). Without a root yet, or after a watch
 * error, the watch is retried every 5 s.
 *
 * Ported from omp-web `lib/session-watcher.ts` (MIT, Copyright (c) 2026 agegr, https://github.com/ompweb-dev/ompweb).
 */
import { watch, type FSWatcher } from "node:fs";
import { join } from "node:path";
import { sessionsRoot } from "./paths.ts";

export const DEBOUNCE_MS = 500;
const RETRY_MS = 5_000;

/** Watch the live session root until the returned function is called. */
export function watchSessions(onChange: (sessionPaths: string[]) => void): () => void {
  let watcher: FSWatcher | undefined;
  let flushTimer: Timer | undefined;
  let retryTimer: Timer | undefined;
  let pending = new Set<string>();
  let unknown = false;
  let stopped = false;

  const flush = (): void => {
    flushTimer = undefined;
    const paths = unknown ? [] : [...pending];
    pending = new Set();
    unknown = false;
    onChange(paths);
  };

  const record = (root: string, filename: string | null): void => {
    if (filename === null) unknown = true;
    else if (filename.endsWith(".jsonl")) pending.add(join(root, filename));
    else return;
    flushTimer ??= setTimeout(flush, DEBOUNCE_MS);
  };

  const start = (): void => {
    retryTimer = undefined;
    if (stopped) return;
    const root = sessionsRoot();
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
    clearTimeout(flushTimer);
    clearTimeout(retryTimer);
    watcher?.close();
  };
}
