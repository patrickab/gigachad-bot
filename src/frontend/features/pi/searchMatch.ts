/**
 * Where a session search's query matches in a text: the browser's copy of pi-host's `findSearchMatches`
 * (`pi-host/src/host/local/query.ts`, M24), so the matches painted in the transcript are the ones the host counted.
 * A hit's `ordinal` is its index among an entry's spans in text order. `__tests__/searchMatch.test.ts` keeps the two equal.
 *
 * Unquoted words match fuzzily and in any order, "quoted text" verbatim; one matching term is enough for a span.
 */

// \p{M}: combining marks belong to their word (Devanagari vowel signs, NFD accents).
const WORD_RE = /[\p{L}\p{M}\p{N}]+/gu

function normalize(text: string): string {
  return text.normalize("NFC").toLowerCase()
}

export interface SearchQuery {
  terms: { text: string; quality: (word: string) => number }[]
  phrases: string[]
}

export interface SearchSpan { start: number; end: number; quality: number }

/** Levenshtein distance of `a` and `b`, `max + 1` once it exceeds `max`. */
function editDistance(a: string, b: string, max: number): number {
  if (Math.abs(a.length - b.length) > max) return max + 1
  let previous = Array.from({ length: b.length + 1 }, (_, index) => index)
  let current = new Array<number>(b.length + 1).fill(0)
  for (let i = 1; i <= a.length; i++) {
    current[0] = i
    let rowMin = i
    for (let j = 1; j <= b.length; j++) {
      current[j] = Math.min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1))
      rowMin = Math.min(rowMin, current[j])
    }
    if (rowMin > max) return max + 1;
    [previous, current] = [current, previous]
  }
  return previous[b.length]
}

/** Word-match quality ladder: exact > prefix > substring > 1 typo > 2 typos. */
function termQuality(term: string, word: string): number {
  if (word === term) return 1
  if (word.startsWith(term)) return 0.8
  if (word.includes(term)) return 0.6
  const maxTypos = term.length >= 8 ? 2 : term.length >= 5 ? 1 : 0
  if (maxTypos === 0) return 0
  const fullDistance = editDistance(term, word, maxTypos)
  const distance = word.length <= term.length || fullDistance <= 1
    ? fullDistance
    : Math.min(fullDistance, editDistance(term, word.slice(0, term.length), maxTypos))
  if (distance > maxTypos) return 0
  return distance === 1 ? 0.45 : 0.3
}

/** Terms and phrases of `raw`, null when it has neither. */
export function compileSearchQuery(raw: string): SearchQuery | null {
  const terms: string[] = []
  const phrases: string[] = []
  // An unclosed quote is a phrase still being typed.
  for (const match of normalize(raw).matchAll(/["“”„«»]([^"“”„«»]*)["“”„«»]?|[^\s"“”„«»]+/g)) {
    if (match[1] !== undefined) {
      const phrase = match[1].replace(/\s+/g, " ").trim()
      if (phrase) phrases.push(phrase)
    } else {
      terms.push(...(match[0].match(WORD_RE) ?? []))
    }
  }
  if (terms.length === 0 && phrases.length === 0) return null
  return {
    phrases,
    terms: terms.map((text) => {
      const memo = new Map<string, number>()
      return {
        text,
        quality: (word: string) => {
          let quality = memo.get(word)
          if (quality === undefined) memo.set(word, (quality = termQuality(text, word)))
          return quality
        },
      }
    }),
  }
}

/** Every span of `text` the query matches, in text order and never overlapping. */
export function findSearchMatches(text: string, query: SearchQuery): SearchSpan[] {
  const spans: SearchSpan[] = []
  for (const phrase of query.phrases) {
    const pattern = phrase.split(" ").map((part) => part.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("\\s+")
    for (const match of text.matchAll(new RegExp(pattern, "giu"))) {
      spans.push({ start: match.index, end: match.index + match[0].length, quality: 1 })
    }
  }
  if (query.terms.length > 0) {
    for (const match of text.matchAll(WORD_RE)) {
      const word = normalize(match[0])
      let best = 0
      let bestTerm = ""
      for (const term of query.terms) {
        const quality = term.quality(word)
        if (quality > best) {
          best = quality
          bestTerm = term.text
        }
      }
      if (best === 0) continue
      let start = match.index
      let end = start + match[0].length
      // Exact, prefix and substring hits mark only the typed term; typo hits mark the whole word.
      const at = best >= 0.6 && word.length === match[0].length ? word.indexOf(bestTerm) : -1
      if (at >= 0) {
        start += at
        end = start + bestTerm.length
      }
      spans.push({ start, end, quality: best })
    }
  }
  spans.sort((a, b) => a.start - b.start || b.end - a.end)
  const merged: SearchSpan[] = []
  for (const span of spans) if (span.start >= (merged.at(-1)?.end ?? 0)) merged.push(span)
  return merged
}
