export interface PatchLine {
  kind: "hunk" | "added" | "removed" | "context" | "note"
  text: string
  oldNumber: number | null
  newNumber: number | null
}

const HUNK_HEADER = /^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/

/** Display lines of a unified patch with line numbers, file headers before the first hunk dropped. */
export function parsePatch(patch: string): PatchLine[] {
  const lines: PatchLine[] = []
  let oldNumber = 0
  let newNumber = 0
  let inHunk = false
  for (const text of patch.split("\n")) {
    const header = HUNK_HEADER.exec(text)
    if (header) {
      oldNumber = Number(header[1])
      newNumber = Number(header[2])
      inHunk = true
      lines.push({ kind: "hunk", text, oldNumber: null, newNumber: null })
    } else if (!inHunk) {
      continue
    } else if (text.startsWith("\\")) {
      lines.push({ kind: "note", text, oldNumber: null, newNumber: null })
    } else if (text.startsWith("+")) {
      lines.push({ kind: "added", text: text.slice(1), oldNumber: null, newNumber: newNumber++ })
    } else if (text.startsWith("-")) {
      lines.push({ kind: "removed", text: text.slice(1), oldNumber: oldNumber++, newNumber: null })
    } else if (text.startsWith(" ")) {
      lines.push({ kind: "context", text: text.slice(1), oldNumber: oldNumber++, newNumber: newNumber++ })
    }
  }
  return lines
}

/** Patch indices per split row, full-width for headers and patch notes. */
export type SplitRow = { full: number } | { old: number | null; new: number | null }

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
