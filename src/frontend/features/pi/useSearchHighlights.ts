"use client"

import { type RefObject } from "react"
import { currentRange as textCurrentRange, useTextHighlights, type TextHighlight, type TextMatcher } from "@/features/conversation/useTextHighlights"
import type { PiSearchFocus } from "./searchFocus"
import { compileSearchQuery, findSearchMatches } from "./searchMatch"

/** The transcript elements holding a message's text, tagged with its entry id. */
export const ENTRY_TEXT = "[data-entry-id]"

/** Matches as pi-host counts them, see `searchMatch.ts`. */
const piMatcher: TextMatcher = (raw) => {
  const query = compileSearchQuery(raw)
  return query && ((text) => findSearchMatches(text, query))
}

/** The search as the generic highlight, null without a shown hit. */
function piHighlight(search: PiSearchFocus | null): TextHighlight | null {
  const hit = search?.matches[search.index]
  if (!search || !hit) return null
  return { name: "pi-search", query: search.query, matcher: piMatcher, attribute: "data-entry-id", current: { value: hit.entryId, ordinal: hit.ordinal } }
}

/** The shown hit's range, else null (its text is not rendered the way the host counted it). */
export function currentRange(container: Element, search: PiSearchFocus): Range | null {
  const highlight = piHighlight(search)
  return highlight && textCurrentRange(container, highlight)
}

/** Paints every hit of the search in the mounted transcript and the shown one apart. Cleared with the search. */
export function useSearchHighlights(container: RefObject<HTMLElement | null>, search: PiSearchFocus | null): void {
  useTextHighlights(container, piHighlight(search))
}
