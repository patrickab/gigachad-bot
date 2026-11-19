export interface PatchLine {
  kind: "hunk" | "added" | "removed" | "context" | "note"
  text: string
  oldNumber: number | null
  newNumber: number | null
}

/** One file of a unified patch: paths from its `---`/`+++` header (absent for a bare hunk list), its numbered lines. */
export interface PatchFile {
  oldPath?: string
  newPath?: string
  lines: PatchLine[]
}

const HUNK_HEADER = /^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/

/** File path of a `---`/`+++` header without its trailing timestamp. */
function cleanPatchPath(path: string): string {
  return path.split("\t")[0].trim()
}

/**
 * The files of a unified patch with numbered display lines, the one parser for every diff view (K5).
 *
 * - A `--- ` line followed by `+++ ` starts a file, a hunk header before any starts a pathless one
 * - Lines before a file's first hunk (`diff --git`, `index`) and lines no hunk prefix matches are dropped
 * - `\ No newline at end of file` becomes a `note`
 */
export function parsePatchFiles(patch: string): PatchFile[] {
  const files: PatchFile[] = []
  const lines = patch.split(/\r?\n/)
  let current: PatchFile | null = null
  let oldNumber = 0
  let newNumber = 0
  let inHunk = false
  for (let index = 0; index < lines.length; index++) {
    const text = lines[index]
    if (text.startsWith("--- ") && lines[index + 1]?.startsWith("+++ ")) {
      current = { oldPath: cleanPatchPath(text.slice(4)), newPath: cleanPatchPath(lines[++index].slice(4)), lines: [] }
      files.push(current)
      inHunk = false
      continue
    }
    const header = HUNK_HEADER.exec(text)
    if (header) {
      if (!current) {
        current = { lines: [] }
        files.push(current)
      }
      oldNumber = Number(header[1])
      newNumber = Number(header[2])
      inHunk = true
      current.lines.push({ kind: "hunk", text, oldNumber: null, newNumber: null })
    } else if (!inHunk || !current) {
      continue
    } else if (text.startsWith("\\")) {
      current.lines.push({ kind: "note", text, oldNumber: null, newNumber: null })
    } else if (text.startsWith("+")) {
      current.lines.push({ kind: "added", text: text.slice(1), oldNumber: null, newNumber: newNumber++ })
    } else if (text.startsWith("-")) {
      current.lines.push({ kind: "removed", text: text.slice(1), oldNumber: oldNumber++, newNumber: null })
    } else if (text.startsWith(" ")) {
      current.lines.push({ kind: "context", text: text.slice(1), oldNumber: oldNumber++, newNumber: newNumber++ })
    }
  }
  return files
}

/** Display lines of a unified patch with line numbers, every file's lines in order, file headers dropped. */
export function parsePatch(patch: string): PatchLine[] {
  return parsePatchFiles(patch).flatMap((file) => file.lines)
}

/** Patch indices per split row, full-width for headers and patch notes. */
type SplitRow = { full: number } | { old: number | null; new: number | null }

/** Pair each run of removed lines with the added run after it, context lines appear on both sides. */
export function toSplitRows(lines: PatchLine[]): SplitRow[] {
  const rows: SplitRow[] = []
  let index = 0
  while (index < lines.length) {
    const kind = lines[index]!.kind
    if (kind === "hunk" || kind === "note") rows.push({ full: index++ })
    else if (kind === "context") rows.push({ old: index, new: index++ })
    else {
      const removed: number[] = []
      const added: number[] = []
      while (lines[index]?.kind === "removed") removed.push(index++)
      while (lines[index]?.kind === "added") added.push(index++)
      for (let pair = 0; pair < Math.max(removed.length, added.length); pair++) rows.push({ old: removed[pair] ?? null, new: added[pair] ?? null })
    }
  }
  return rows
}
