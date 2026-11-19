/**
 * Session search scoring (M24): query compilation, weighted field scores and match spans for snippets.
 *
 * - Consumes: raw query text and the text of a session's fields
 * - Provides: `compileSearchQuery`, `searchField`, `scoreSearchFields`, `findSearchMatches`, the field weights
 *
 * Unquoted words match fuzzily and in any order (every word must match somewhere); keeping the typed order scores
 * higher. "Quoted text" (straight or typographic quotes) must appear verbatim, case-insensitive. Scripts written
 * without spaces (CJK) form one word per run, so they match by substring only.
 *
 * Ported from omp-web `lib/session-search.ts` (MIT, Copyright (c) 2026 agegr, https://github.com/ompweb-dev/ompweb).
 */

/** A session's name (`/name`) outranks transcript text. */
export const TITLE_WEIGHT = 100;
export const CONTENT_WEIGHT = 1;

// \p{M}: combining marks belong to their word (Devanagari vowel signs, NFD accents).
const WORD_RE = /[\p{L}\p{M}\p{N}]+/gu;

function normalize(text: string): string {
  return text.normalize("NFC").toLowerCase();
}

export type SearchField = {
  /** Lowercased, whitespace-collapsed text (phrase and order checks). */
  text: string;
  /** Unique lowercased words (term matching). */
  words: string[];
  weight: number;
};

export type CompiledSearchQuery = {
  terms: { text: string; quality: (word: string) => number }[];
  phrases: string[];
};

export type SearchMatchSpan = {
  start: number;
  end: number;
  /** `termQuality`'s ladder (1 exact … 0.3 two typos); quoted phrases count as exact. */
  quality: number;
};

export function searchField(text: string, weight: number): SearchField {
  const normalized = normalize(text).replace(/\s+/g, " ").trim();
  return { text: normalized, words: [...new Set(normalized.match(WORD_RE) ?? [])], weight };
}

/** Levenshtein distance of `a` and `b`, `max + 1` once it exceeds `max`. */
function editDistance(a: string, b: string, max: number): number {
  if (Math.abs(a.length - b.length) > max) return max + 1;
  let previous = Array.from({ length: b.length + 1 }, (_, index) => index);
  let current = new Array<number>(b.length + 1).fill(0);
  for (let i = 1; i <= a.length; i++) {
    current[0] = i;
    let rowMin = i;
    for (let j = 1; j <= b.length; j++) {
      current[j] = Math.min(previous[j]! + 1, current[j - 1]! + 1, previous[j - 1]! + (a[i - 1] === b[j - 1] ? 0 : 1));
      rowMin = Math.min(rowMin, current[j]!);
    }
    if (rowMin > max) return max + 1;
    [previous, current] = [current, previous];
  }
  return previous[b.length]!;
}

/** Word-match quality ladder: exact > prefix > substring > 1 typo > 2 typos. */
function termQuality(term: string, word: string): number {
  if (word === term) return 1;
  if (word.startsWith(term)) return 0.8;
  if (word.includes(term)) return 0.6;
  const maxTypos = term.length >= 8 ? 2 : term.length >= 5 ? 1 : 0;
  if (maxTypos === 0) return 0;
  // The prefix comparison tolerates a typo in a word still being typed.
  const fullDistance = editDistance(term, word, maxTypos);
  const distance =
    word.length <= term.length || fullDistance <= 1
      ? fullDistance
      : Math.min(fullDistance, editDistance(term, word.slice(0, term.length), maxTypos));
  if (distance > maxTypos) return 0;
  return distance === 1 ? 0.45 : 0.3;
}

/** Terms and phrases of `raw`, null when it has neither. */
export function compileSearchQuery(raw: string): CompiledSearchQuery | null {
  const terms: string[] = [];
  const phrases: string[] = [];
  // An unclosed quote is a phrase still being typed.
  for (const match of normalize(raw).matchAll(/["“”„«»]([^"“”„«»]*)["“”„«»]?|[^\s"“”„«»]+/g)) {
    if (match[1] !== undefined) {
      const phrase = match[1].replace(/\s+/g, " ").trim();
      if (phrase) phrases.push(phrase);
    } else {
      terms.push(...(match[0].match(WORD_RE) ?? []));
    }
  }
  if (terms.length === 0 && phrases.length === 0) return null;
  return {
    phrases,
    terms: terms.map((text) => {
      // Memoized per query: the same vocabulary repeats across sessions.
      const memo = new Map<string, number>();
      return {
        text,
        quality: (word: string) => {
          let quality = memo.get(word);
          if (quality === undefined) memo.set(word, (quality = termQuality(text, word)));
          return quality;
        },
      };
    }),
  };
}

function inOrder(text: string, terms: string[]): boolean {
  let from = 0;
  for (const term of terms) {
    const at = text.indexOf(term, from);
    if (at < 0) return false;
    from = at + term.length;
  }
  return true;
}

/** Score of a document split into weighted fields, null when it does not match. */
export function scoreSearchFields(fields: SearchField[], query: CompiledSearchQuery): number | null {
  let score = 0;
  for (const phrase of query.phrases) {
    let best = 0;
    for (const field of fields) if (field.weight > best && field.text.includes(phrase)) best = field.weight;
    if (best === 0) return null;
    score += best;
  }
  for (const term of query.terms) {
    let best = 0;
    for (const field of fields) {
      if (field.weight <= best) continue;
      let quality = 0;
      for (const word of field.words) {
        quality = Math.max(quality, term.quality(word));
        if (quality === 1) break;
      }
      best = Math.max(best, quality * field.weight);
    }
    if (best === 0) return null;
    score += best;
  }
  if (query.terms.length > 1) {
    const texts = query.terms.map((term) => term.text);
    const adjacent = texts.join(" ");
    let bonus = 0;
    for (const field of fields) {
      if (field.text.includes(adjacent)) bonus = Math.max(bonus, field.weight);
      else if (inOrder(field.text, texts)) bonus = Math.max(bonus, field.weight / 2);
    }
    score += bonus;
  }
  return score;
}

/**
 * Every span of `text` the query matches, in text order and never overlapping: each quoted phrase verbatim, and
 * each word that matches a term, fuzzy hits included. Unlike `scoreSearchFields`, one matching term is enough.
 */
export function findSearchMatches(text: string, query: CompiledSearchQuery): SearchMatchSpan[] {
  const spans: SearchMatchSpan[] = [];
  for (const phrase of query.phrases) {
    const pattern = phrase
      .split(" ")
      .map((part) => part.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
      .join("\\s+");
    for (const match of text.matchAll(new RegExp(pattern, "giu"))) {
      spans.push({ start: match.index, end: match.index + match[0].length, quality: 1 });
    }
  }
  if (query.terms.length > 0) {
    for (const match of text.matchAll(WORD_RE)) {
      const word = normalize(match[0]);
      let best = 0;
      let bestTerm = "";
      for (const term of query.terms) {
        const quality = term.quality(word);
        if (quality > best) {
          best = quality;
          bestTerm = term.text;
        }
      }
      if (best === 0) continue;
      let start = match.index;
      let end = start + match[0].length;
      // Exact, prefix and substring hits mark only the typed term; typo hits mark the whole word.
      const at = best >= 0.6 && word.length === match[0].length ? word.indexOf(bestTerm) : -1;
      if (at >= 0) {
        start += at;
        end = start + bestTerm.length;
      }
      spans.push({ start, end, quality: best });
    }
  }
  spans.sort((a, b) => a.start - b.start || b.end - a.end);
  const merged: SearchMatchSpan[] = [];
  for (const span of spans) if (span.start >= (merged.at(-1)?.end ?? 0)) merged.push(span);
  return merged;
}
