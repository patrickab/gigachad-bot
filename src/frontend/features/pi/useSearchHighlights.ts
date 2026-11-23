"use client"

import { useEffect, type RefObject } from "react"
import type { PiSearchFocus } from "./searchFocus"
import { compileSearchQuery, findSearchMatches, type SearchQuery } from "./searchMatch"

const ALL_HITS = "pi-search"
const CURRENT_HIT = "pi-search-current"
/** The transcript elements holding a message's text, tagged with its entry id. */
export const ENTRY_TEXT = "[data-entry-id]"
/** Streaming and re-renders replace text nodes: repaint at most this often. */
const PAINT_DELAY_MS = 150
/** Rendered text the host's count does not see: buttons (code headers) and KaTeX's output (math is painted twice). */
const NOT_MESSAGE_TEXT = "button, .katex"

/** The CSS Custom Highlight registry, null in a browser without it: hits are then not painted, jumping to one still works. */
function highlightRegistry(): HighlightRegistry | null {
  return typeof CSS !== "undefined" && "highlights" in CSS && typeof Highlight !== "undefined" ? CSS.highlights : null
}

/** Every hit inside `element`, in text order (a hit's index is its host ordinal). Matched per text node, as markup splits phrases. */
// shortcut: markdown the host counts as raw text (link targets, a phrase split by formatting) shifts the later ordinals of that message, the jump then lands near it.
export function elementHits(element: Element, query: SearchQuery): Range[] {
  const ranges: Range[] = []
  const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT, {
    acceptNode: (node) => (node.parentElement?.closest(NOT_MESSAGE_TEXT) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
  })
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    for (const span of findSearchMatches(node.nodeValue ?? "", query)) {
      const range = document.createRange()
      range.setStart(node, span.start)
      range.setEnd(node, span.end)
      ranges.push(range)
    }
  }
  return ranges
}

/** The elements of entry `entryId`, in document order (an answer's text blocks). */
export function entryElements(container: Element, entryId: string): Element[] {
  return [...container.querySelectorAll(`${ENTRY_TEXT}`)].filter((element) => element.getAttribute("data-entry-id") === entryId)
}

/** The shown hit's range, else null (its text is not rendered the way the host counted it). */
export function currentRange(container: Element, search: PiSearchFocus): Range | null {
  const query = compileSearchQuery(search.query)
  const hit = search.matches[search.index]
  if (!query || !hit) return null
  return entryElements(container, hit.entryId).flatMap((element) => elementHits(element, query))[hit.ordinal] ?? null
}

/**
 * Paints every hit of the search in the mounted transcript and the shown one apart, with the CSS Custom Highlight API
 * (the DOM stays untouched). Cleared with the search.
 */
export function useSearchHighlights(container: RefObject<HTMLElement | null>, search: PiSearchFocus | null): void {
  const query = search?.query
  const entryId = search?.matches[search.index]?.entryId
  const ordinal = search?.matches[search.index]?.ordinal
  useEffect(() => {
    const element = container.current
    const registry = highlightRegistry()
    const compiled = query ? compileSearchQuery(query) : null
    if (!element || !registry || !compiled || entryId === undefined || ordinal === undefined) return
    let timer: ReturnType<typeof setTimeout> | undefined
    const paint = () => {
      timer = undefined
      const all = new Highlight()
      for (const text of element.querySelectorAll(ENTRY_TEXT)) for (const range of elementHits(text, compiled)) all.add(range)
      registry.set(ALL_HITS, all)
      const range = entryElements(element, entryId).flatMap((text) => elementHits(text, compiled))[ordinal]
      if (range) registry.set(CURRENT_HIT, new Highlight(range))
      else registry.delete(CURRENT_HIT)
    }
    const schedule = () => { timer ??= setTimeout(paint, PAINT_DELAY_MS) }
    paint()
    const observer = new MutationObserver(schedule)
    observer.observe(element, { childList: true, subtree: true, characterData: true })
    return () => {
      observer.disconnect()
      clearTimeout(timer)
      registry.delete(ALL_HITS)
      registry.delete(CURRENT_HIT)
    }
  }, [container, query, entryId, ordinal])
}
