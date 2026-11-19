import { baseName, relativeIn } from "@/lib/paths"

/** The line that names the file at `path` to a new session in `cwd`: relative to `cwd` when inside it, else absolute. */
export function fileLead(path: string, cwd: string): string {
  return `File: ${relativeIn(cwd, path)}`
}

/** The line that names a LaTeX project to a new session in `cwd`: its folder (relative to `cwd` when inside it, else absolute) and its root file relative to the folder. */
export function projectLead(folder: string, root: string | null, cwd: string): string {
  const lead = `Project: ${relativeIn(cwd, folder)}`
  return root === null ? lead : `${lead}, root: ${relativeIn(folder, root)}`
}

/** The line that names canvas `key` to a new session: its file name. The session sees the canvas through screenshots only. */
export function canvasLead(key: string): string {
  return `Canvas: ${baseName(key)}`
}

/** `text` without the `File:`, `Project:` or `Canvas:` line a lead put in front of a first message, so session titles show the request. */
export function withoutFileLead(text: string): string {
  return /^(?:File|Project|Canvas): /.test(text) ? text.replace(/^(?:File|Project|Canvas): [^\n]*\n*/, "") : text
}
