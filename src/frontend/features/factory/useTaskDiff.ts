// A task's changes. Consumes `getFactoryTaskDiff` (one unified diff of the task against its base), provides
// `parseTaskDiff` (the diff as files the diff components render) and `useTaskDiff` (the files, refetched on demand).
import { useEffect, useState } from "react"
import { ApiError, getFactoryTaskDiff } from "@/lib/backend/api"
import type { FilePatch } from "@/features/editor/diff/useRootDiff"
import type { DiffStatus } from "@/lib/domain/files"
import { parsePatch } from "@/lib/diff"
import { errorMessage } from "@/lib/utils"

/** One file of a task's diff, shaped like a changed file of a root diff. */
export interface TaskDiffFile {
  path: string
  status: DiffStatus
  /** Line counts, null for a file without a text diff (binary, mode change). */
  added: number | null
  removed: number | null
  /** The file's section of the diff text: an unchanged section keeps its parsed lines across refetches. */
  stamp: string
  patch: FilePatch
}

const FILE_HEADER = "diff --git "

/** Path after an `a/` or `b/` prefix of a `---`, `+++` or rename line, undefined for `/dev/null`. */
function headerPath(section: string, prefix: string): string | undefined {
  const line = section.split("\n").find((candidate) => candidate.startsWith(prefix))
  const path = line?.slice(prefix.length).split("\t")[0]
  return path === undefined || path === "/dev/null" ? undefined : path.replace(/^[ab]\//, "")
}

/** A section's path: the new side, else the old one, else one half of `diff --git a/<path> b/<path>` (a binary file names no sides). */
function sectionPath(section: string): string {
  const named = headerPath(section, "+++ ") ?? headerPath(section, "--- ") ?? headerPath(section, "rename to ")
  if (named !== undefined) return named
  const both = section.split("\n", 1)[0].slice(FILE_HEADER.length)
  return both.slice("a/".length, (both.length - 1) / 2)
}

/** Read from the lines before the first hunk, where git names what happened to the file. */
function sectionStatus(section: string): DiffStatus {
  const header = section.split("\n@@", 1)[0]
  if (/^new file mode /m.test(header)) return "added"
  if (/^deleted file mode /m.test(header)) return "deleted"
  if (/^rename from /m.test(header)) return "renamed"
  return "modified"
}

function parseSection(section: string): TaskDiffFile {
  const parsed = parsePatch(section)
  const lines = parsed.length > 0 ? parsed : null
  const count = (kind: "added" | "removed") => lines ? lines.filter((line) => line.kind === kind).length : null
  return { path: sectionPath(section), status: sectionStatus(section), added: count("added"), removed: count("removed"), stamp: section, patch: { lines, failed: false, stamp: section } }
}

/**
 * The files of a `git diff` text, in its order. A file whose section equals the one in `previous` is that same object,
 * so its rendered lines are not rebuilt.
 */
export function parseTaskDiff(diff: string, previous: readonly TaskDiffFile[] = []): TaskDiffFile[] {
  const known = new Map(previous.map((file) => [file.stamp, file]))
  return diff
    .split(/^(?=diff --git )/m)
    .filter((section) => section.startsWith(FILE_HEADER))
    .map((section) => known.get(section) ?? parseSection(section))
}

/**
 * The files of task `taskId`'s diff, undefined until the first answer.
 *
 * - Read on mount and again whenever `revision` changes, the shown files stay until the new ones arrive
 * - A task without a diff base yet (404, it is still starting) has no files and no error
 */
export function useTaskDiff(taskId: string, revision: number): { files: TaskDiffFile[] | undefined; error: string } {
  const [files, setFiles] = useState<TaskDiffFile[]>()
  const [error, setError] = useState("")

  useEffect(() => {
    let cancelled = false
    getFactoryTaskDiff(taskId).then(
      ({ diff }) => {
        if (cancelled) return
        setFiles((shown) => parseTaskDiff(diff, shown))
        setError("")
      },
      (cause: unknown) => {
        if (cancelled) return
        setFiles((shown) => shown ?? [])
        setError(cause instanceof ApiError && cause.status === 404 ? "" : errorMessage(cause, "Could not load the changes"))
      },
    )
    return () => {
      cancelled = true
    }
  }, [taskId, revision])

  return { files, error }
}
