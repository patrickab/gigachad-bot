// Ported from kahme247/ompweb lib/patch.ts (MIT, Copyright (c) 2026 agegr).
import { parsePatchFiles } from "@/lib/diff"

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

/** Split view of a unified patch (`--- a`, `+++ b`, `@@ -1,2 +1,3 @@`) parsed by `lib/diff.ts`, null when the text is no unified patch. */
export function parseUnifiedPatch(text: string): SplitDiffFile[] | null {
  const files = parsePatchFiles(text).map(({ oldPath, newPath, lines }) => {
    const builder = new SplitRowBuilder({ oldPath, newPath, rows: [] })
    for (const line of lines) {
      if (line.kind === "hunk" || line.kind === "note") builder.hunk(line.text)
      else if (line.kind === "context") builder.context(line.oldNumber!, line.newNumber!, line.text)
      else if (line.kind === "removed") builder.remove(line.oldNumber!, line.text)
      else builder.add(line.newNumber!, line.text)
    }
    builder.flush()
    return builder.file
  }).filter((file) => file.rows.some((row) => row.type === "line"))
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
