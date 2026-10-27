import { resolveLocalFileHref } from "../../lib/file-links";

/** Absolute path for the shell's file view: relative tool paths resolve against the session cwd, others pass through. */
export function resolveOpenFilePath(path: string, cwd: string | undefined): string {
  return resolveLocalFileHref(path, cwd) ?? path;
}
