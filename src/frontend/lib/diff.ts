/** Line diff as text, one line per entry prefixed "  " (kept), "+ " (added) or "- " (removed). O(n*m) time and memory. */
export function computeLineDiff(oldText: string, newText: string): string {
  const oldLines = oldText.split("\n")
  const newLines = newText.split("\n")

  // Standard DP for LCS
  const dp: number[][] = Array.from({ length: oldLines.length + 1 }, () =>
    Array(newLines.length + 1).fill(0)
  )

  for (let i = 1; i <= oldLines.length; i++) {
    for (let j = 1; j <= newLines.length; j++) {
      if (oldLines[i - 1] === newLines[j - 1]) {
        dp[i]![j] = dp[i - 1]![j - 1]! + 1
      } else {
        dp[i]![j] = Math.max(dp[i - 1]![j]!, dp[i]![j - 1]!)
      }
    }
  }

  const diffLines: string[] = []
  let i = oldLines.length
  let j = newLines.length

  // Backtrack from the end, then reverse. Ties emit the addition first so removals precede additions in the output.
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && oldLines[i - 1] === newLines[j - 1]) {
      diffLines.push(`  ${oldLines[i - 1]}`)
      i--
      j--
    } else if (j > 0 && (i === 0 || dp[i]![j - 1]! >= dp[i - 1]![j]!)) {
      diffLines.push(`+ ${newLines[j - 1]}`)
      j--
    } else {
      diffLines.push(`- ${oldLines[i - 1]}`)
      i--
    }
  }

  return diffLines.reverse().join("\n")
}

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
