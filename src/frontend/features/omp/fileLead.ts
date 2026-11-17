import { relativeIn } from "@/lib/paths"

/** The line that names the file at `path` to a new session in `cwd`: relative to `cwd` when inside it, else absolute. */
export function fileLead(path: string, cwd: string): string {
  return `File: ${relativeIn(cwd, path)}`
}

/**
 * The line that names a LaTeX project to a new session in `cwd`: its folder (relative to `cwd` when inside it, else absolute) and its root file relative to the folder.
 *
 * - `guide` is where the instructions live when `cwd` does not hold them (a mount): the line tells the session to read them first
 */
export function projectLead(folder: string, root: string | null, cwd: string, guide?: string | null): string {
  const lead = `Project: ${relativeIn(cwd, folder)}`
  const named = root === null ? lead : `${lead}, root: ${relativeIn(folder, root)}`
  return guide ? `${named}. Instructions: read ${guide}/.omp/SYSTEM.md first, the template sets are in ${guide}/.templates/` : named
}

/** `text` without the `File:` or `Project:` line a lead put in front of a first message, so session titles show the request. */
export function withoutFileLead(text: string): string {
  return /^(?:File|Project): /.test(text) ? text.replace(/^(?:File|Project): [^\n]*\n*/, "") : text
}
