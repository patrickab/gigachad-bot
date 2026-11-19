// Ported from kahme247/ompweb lib/patch.ts (MIT, Copyright (c) 2026 agegr).

export type SplitDiffCellType = "context" | "removed" | "added" | "empty"

export interface SplitDiffCell { lineNo: number | null; text: string; type: SplitDiffCellType }

export type SplitDiffRow =
  | { type: "hunk"; text: string }
  | { type: "line"; left: SplitDiffCell; right: SplitDiffCell }

export interface SplitDiffFile { oldPath?: string; newPath?: string; rows: SplitDiffRow[] }

interface PendingChangeLine { lineNo: number; text: string }

/** Collects removed/added runs and pairs them side by side when a context line or hunk ends the run. */
class SplitRowBuilder {
  private removed: PendingChangeLine[] = []
  private added: PendingChangeLine[] = []

  constructor(readonly file: SplitDiffFile) {}

  remove(lineNo: number, text: string) { this.removed.push({ lineNo, text }) }

  add(lineNo: number, text: string) { this.added.push({ lineNo, text }) }

  flush() {
    const count = Math.max(this.removed.length, this.added.length)
    for (let i = 0; i < count; i++) {
      const left = this.removed[i] ? { ...this.removed[i], type: "removed" as const } : EMPTY_CELL
      const right = this.added[i] ? { ...this.added[i], type: "added" as const } : EMPTY_CELL
      this.file.rows.push({ type: "line", left, right })
    }
    this.removed = []
    this.added = []
  }

  context(oldLineNo: number, newLineNo: number, text: string) {
    this.flush()
    this.file.rows.push({ type: "line", left: { lineNo: oldLineNo, text, type: "context" }, right: { lineNo: newLineNo, text, type: "context" } })
  }

  hunk(text: string) {
    this.flush()
    this.file.rows.push({ type: "hunk", text })
  }
}

const EMPTY_CELL: SplitDiffCell = { lineNo: null, text: "", type: "empty" }

/** Split view of a unified patch (`--- a`, `+++ b`, `@@ -1,2 +1,3 @@`), null when the text is no unified patch. */
export function parseUnifiedPatch(text: string): SplitDiffFile[] | null {
  const builders: SplitRowBuilder[] = []
  let pendingOldPath: string | undefined
  let oldLineNo = 0
  let newLineNo = 0

  for (const line of text.split(/\r?\n/)) {
    let current = builders.at(-1)
    if (line.startsWith("--- ")) {
      current?.flush()
      pendingOldPath = cleanPatchPath(line.slice(4))
      continue
    }
    if (line.startsWith("+++ ")) {
      current?.flush()
      builders.push(new SplitRowBuilder({ oldPath: pendingOldPath, newPath: cleanPatchPath(line.slice(4)), rows: [] }))
      continue
    }
    const hunk = line.match(/^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/)
    if (hunk) {
      if (!current) {
        current = new SplitRowBuilder({ rows: [] })
        builders.push(current)
      }
      oldLineNo = Number(hunk[1])
      newLineNo = Number(hunk[2])
      current.hunk(line)
      continue
    }
    if (!current) continue

    const prefix = line[0]
    const content = line.slice(1)
    if (line.startsWith("\\ ")) current.hunk(line)
    else if (prefix === " ") current.context(oldLineNo++, newLineNo++, content)
    else if (prefix === "-") current.remove(oldLineNo++, content)
    else if (prefix === "+") current.add(newLineNo++, content)
    else if (line !== "") current.hunk(line)
  }
  builders.at(-1)?.flush()

  const files = builders.map((builder) => builder.file).filter((file) => file.rows.some((row) => row.type === "line"))
  return files.length > 0 ? files : null
}

/**
 * Split view of Pi's display diff (`details.diff` of the `edit` tool), null when the text is not one.
 *
 * - Lines are `<+|-| ><padded line number> <text>`, a skipped gap is ` <padding> ...`
 * - Context lines carry the old line number only, the new one is derived from the running offset
 */
export function parsePiDisplayDiff(text: string): SplitDiffFile[] | null {
  const builder = new SplitRowBuilder({ rows: [] })
  let offset = 0

  for (const line of text.split(/\r?\n/)) {
    if (line === "") continue
    if (/^ +\.\.\.$/.test(line)) {
      builder.hunk(line.trim())
      continue
    }
    const match = line.match(/^([+\- ]) *(\d+) ?(.*)$/)
    if (!match) return null
    const [, prefix, number, content] = match
    const lineNo = Number(number)
    if (prefix === "-") {
      builder.remove(lineNo, content)
      offset--
    } else if (prefix === "+") {
      builder.add(lineNo, content)
      offset++
    } else {
      builder.context(lineNo, lineNo + offset, content)
    }
  }
  builder.flush()
  return builder.file.rows.some((row) => row.type === "line") ? [builder.file] : null
}

/** File path of a `---`/`+++` header without its trailing timestamp. */
function cleanPatchPath(path: string): string {
  return path.split("\t")[0].trim()
}
