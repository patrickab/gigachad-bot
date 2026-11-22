// A task's changes. Consumes the core's `tasks.diff` (its changed files with line counts and one unified diff against
// its base), provides `parseTaskDiff` (the core's files with their diff sections, as the diff components render them)
// and `useTaskDiff` (the files, refetched on demand).
import { useEffect, useState } from "react"
import { callFactory } from "@/lib/backend/api"
import type { FactoryTaskDiff } from "@/lib/domain/factory"
import type { FilePatch } from "@/features/editor/diff/useRootDiff"
import type { DiffStatus } from "@/lib/domain/files"
import { parsePatch } from "@/lib/diff"
import { errorMessage } from "@/lib/utils"

/** One file of a task's diff, shaped like a changed file of a root diff. */
export interface TaskDiffFile {
  path: string
  status: DiffStatus
  /** Line counts as the core counts them (0 for a binary file). */
  added: number | null
  removed: number | null
  /** The file's section of the diff text, "" when the diff was cut before it: an unchanged file keeps its parsed lines across refetches. */
  stamp: string
  patch: FilePatch
}

const FILE_HEADER = "diff --git "

/** Path after an `a/` or `b/` prefix of a `---` or `+++` line, undefined for `/dev/null`. */
function headerPath(section: string, prefix: string): string | undefined {
  const line = section.split("\n").find((candidate) => candidate.startsWith(prefix))
  const path = line?.slice(prefix.length).split("\t")[0]
  return path === undefined || path === "/dev/null" ? undefined : path.replace(/^[ab]\//, "")
}

/** A section's path: the new side, else the old one, else one half of `diff --git a/<path> b/<path>` (a binary file names no sides). */
function sectionPath(section: string): string {
  const named = headerPath(section, "+++ ") ?? headerPath(section, "--- ")
  if (named !== undefined) return named
  const both = section.split("\n", 1)[0].slice(FILE_HEADER.length)
  return both.slice("a/".length, (both.length - 1) / 2)
}

/** Read from the lines before the first hunk, where git names what happened to the file. */
function sectionStatus(section: string): DiffStatus {
  const header = section.split("\n@@", 1)[0]
  if (/^new file mode /m.test(header)) return "added"
  if (/^deleted file mode /m.test(header)) return "deleted"
  return "modified"
}

function fileOf({ path, added, removed }: FactoryTaskDiff["files"][number], section: string | undefined): TaskDiffFile {
  const parsed = section === undefined ? [] : parsePatch(section)
  const lines = parsed.length > 0 ? parsed : null
  const stamp = section ?? ""
  return { path, status: section === undefined ? "modified" : sectionStatus(section), added, removed, stamp, patch: { lines, failed: false, stamp } }
}

/**
 * The core's `files` of a `tasks.diff` answer, in its order, each with its section of the diff (the core takes the
 * diff without renames, so one path names one section). A file equal to one in `previous` (path, counts, section) is
 * that same object, so its rendered lines are not rebuilt.
 */
export function parseTaskDiff({ diff, files }: Pick<FactoryTaskDiff, "diff" | "files">, previous: readonly TaskDiffFile[] = []): TaskDiffFile[] {
  const sections = new Map(diff
    .split(/^(?=diff --git )/m)
    .filter((section) => section.startsWith(FILE_HEADER))
    .map((section) => [sectionPath(section), section]))
  const known = new Map(previous.map((file) => [file.path, file]))
  return files.map((file) => {
    const section = sections.get(file.path)
    const shown = known.get(file.path)
    const same = shown && shown.stamp === (section ?? "") && shown.added === file.added && shown.removed === file.removed
    return same ? shown : fileOf(file, section)
  })
}

/**
 * The files of task `taskId`'s diff, undefined until the first answer.
 *
 * - Read on mount and again whenever `revision` changes, the shown files stay until the new ones arrive
 * - A task that has not `started` (no factory state yet) has no diff base: no files, no read, no error
 */
export function useTaskDiff(taskId: string, revision: number, started: boolean): { files: TaskDiffFile[] | undefined; error: string } {
  const [files, setFiles] = useState<TaskDiffFile[]>()
  const [error, setError] = useState("")
  if (!started && files === undefined) setFiles([])

  useEffect(() => {
    if (!started) return
    let cancelled = false
    callFactory<FactoryTaskDiff>("tasks.diff", { id: taskId }).then(
      (answer) => {
        if (cancelled) return
        setFiles((shown) => parseTaskDiff(answer, shown))
        setError("")
      },
      (cause: unknown) => {
        if (cancelled) return
        setFiles((shown) => shown ?? [])
        setError(errorMessage(cause, "Could not load the changes"))
      },
    )
    return () => {
      cancelled = true
    }
  }, [taskId, revision, started])

  return { files, error }
}
