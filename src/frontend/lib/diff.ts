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
