import { extOf } from "@/lib/paths"

/** The file types the workbench opens, one omp workspace each. */
export type WorkbenchKind = "latex" | "markdown"

const WORKBENCH_KINDS: readonly string[] = ["latex", "markdown"] satisfies WorkbenchKind[]

/** The workbench kind of the file at `path` by its extension: `.tex` LaTeX, `.md` Markdown, else null. */
export function workbenchKindOf(path: string): WorkbenchKind | null {
  const ext = extOf(path)
  if (ext === "tex") return "latex"
  if (ext === "md") return "markdown"
  return null
}

/** Whether storage key or disk path `path` opens in the workbench: a `.md` or `.tex`, except notes (`note/<name>`), which stay in the document editor. */
export function opensInWorkbench(path: string): boolean {
  return workbenchKindOf(path) !== null && !path.startsWith("note/")
}

/**
 * Disk path `path` after project `oldSlug` became `newSlug`, else `path` itself.
 *
 * - A project's workbench files live at `<root>/workbench/<kind>/<slug>/<stem>/<file>`, a project rename moves that folder
 * - Only a path of exactly that shape changes, so a root or file merely named like the folders stays
 */
export function relocatedWorkbenchPath(path: string, oldSlug: string, newSlug: string): string {
  const segments = path.split("/")
  const slugAt = segments.length - 3
  const isProjectFile = slugAt >= 2
    && segments[slugAt - 2] === "workbench"
    && WORKBENCH_KINDS.includes(segments[slugAt - 1])
    && segments[slugAt] === oldSlug
  if (!isProjectFile) return path
  segments[slugAt] = newSlug
  return segments.join("/")
}
