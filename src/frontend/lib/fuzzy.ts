import { baseName } from "@/lib/paths"

/**
 * Subsequence fuzzy scorer shared by the file pickers. Returns `null` when the
 * query is not a subsequence of the text; higher scores rank better. Adjacent
 * matches and word-boundary hits (after `/`, space, `-`, `_`) are rewarded, and
 * longer texts are mildly penalised so tighter matches float to the top.
 */
export function fuzzyScore(text: string, query: string): number | null {
  const t = text.toLowerCase()
  const q = query.toLowerCase()
  let ti = 0
  let score = 0
  // Start below -1 so a match at index 0 is not counted as adjacent
  let prev = -2
  for (const ch of q) {
    let found = -1
    for (let k = ti; k < t.length; k++) {
      if (t[k] === ch) { found = k; break }
    }
    if (found === -1) return null
    score += found === prev + 1 ? 3 : 1
    const before = found === 0 ? "/" : t[found - 1]
    if (before === "/" || before === " " || before === "-" || before === "_") score += 2
    prev = found
    ti = found + 1
  }
  return score - text.length * 0.01
}

/** Score of `path` for `query`: a match in `name`, its last segment by default, outranks the same match spread over the folders. */
export function fuzzyPathScore(path: string, query: string, name: string = baseName(path)): number | null {
  const byName = fuzzyScore(name, query)
  const byPath = fuzzyScore(path, query)
  return byName === null ? byPath : Math.max(byName + 1, byPath ?? -Infinity)
}

/** The `items` `score` matches, best first. */
export function rankByScore<T>(items: readonly T[], score: (item: T) => number | null): T[] {
  return items
    .map((item) => ({ item, score: score(item) }))
    .filter((ranked): ranked is { item: T; score: number } => ranked.score !== null)
    .sort((left, right) => right.score - left.score)
    .map(({ item }) => item)
}
