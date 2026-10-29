import { resolveLocalFileHref } from "../../lib/file-links";

/** Absolute path for the shell's file view: relative tool paths resolve against the session cwd, others pass through. */
export function resolveOpenFilePath(path: string, cwd: string | undefined): string {
  return resolveLocalFileHref(path, cwd) ?? path;
}

/** Notice by which an omp extension asks to open a file, e.g. `/files` (`~/.omp/agent/extensions/files.ts`): `Open file: /abs/path`. */
const OPEN_FILE_NOTICE_PREFIX = "Open file: ";

/** Absolute path an open-file notice names, null for any other notice. */
export function openFileNoticePath(message: string): string | null {
  if (!message.startsWith(OPEN_FILE_NOTICE_PREFIX)) return null;
  const path = message.slice(OPEN_FILE_NOTICE_PREFIX.length);
  return path.startsWith("/") && !path.includes("\n") ? path : null;
}
