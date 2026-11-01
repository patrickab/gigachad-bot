import { extOf } from "@/lib/paths"

/** The file types the workbench opens, one omp workspace each. */
export type WorkbenchKind = "latex" | "markdown"

const WORKBENCH_KINDS: readonly string[] = ["latex", "markdown"] satisfies WorkbenchKind[]

const LATEX_EXTENSIONS: readonly string[] = ["tex", "bib", "sty", "cls"]

/** The workbench kind of the file at `path` by its extension: `.tex .bib .sty .cls` LaTeX, `.md` Markdown, else null. */
export function workbenchKindOf(path: string): WorkbenchKind | null {
  const ext = extOf(path)
  if (LATEX_EXTENSIONS.includes(ext)) return "latex"
  if (ext === "md") return "markdown"
  return null
}

/** Whether storage key or disk path `path` opens in the workbench: a `.tex .bib .sty .cls` or `.md` file, except notes (`note/<name>`), which stay in the document editor. */
export function opensInWorkbench(path: string): boolean {
  return workbenchKindOf(path) !== null && !path.startsWith("note/")
}

/**
 * Disk path `path` after project `oldSlug` became `newSlug`, else `path` itself.
 *
 * - A project's workbench files live under `<root>/workbench/<kind>/<slug>/<stem>/`, a project rename moves that folder
 * - Only a path at or below a `<stem>` folder of that shape (or a `<slug>` folder plus one segment, indistinguishable from a folder entry) changes, so a root or file merely named like the folders stays
 */
export function relocatedWorkbenchPath(path: string, oldSlug: string, newSlug: string): string {
  const segments = path.split("/")
  const slugAt = segments.findLastIndex((segment, index) =>
    index >= 2
    && index < segments.length - 1
    && segment === oldSlug
    && segments[index - 2] === "workbench"
    && WORKBENCH_KINDS.includes(segments[index - 1]))
  if (slugAt < 0) return path
  segments[slugAt] = newSlug
  return segments.join("/")
}
