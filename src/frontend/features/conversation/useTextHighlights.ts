"use client"

import { useEffect, type RefObject } from "react"

/** Where a query matches in one text: its spans in text order, never overlapping. */
export type TextFinder = (text: string) => readonly { start: number; end: number }[]
/** Compiles a query into its finder, null for a query without anything to find. A module-level function: an effect keys on it. */
export type TextMatcher = (query: string) => TextFinder | null

/** A query painted in a conversation: every hit in the tagged texts, the shown one apart. */
export interface TextHighlight {
  /** Highlight names: `name` for every hit, `${name}-current` for the shown one (styled in `globals.css`). */
  name: string
  query: string
  matcher: TextMatcher
  /** Attribute tagging the elements whose text is searched; its value names a text (e.g. a message). */
  attribute: string
  /** The shown hit: the `ordinal`-th hit in the elements whose `attribute` is `value`, together in document order. */
  current: { value: string; ordinal: number } | null
}

/** Streaming and re-renders replace text nodes: repaint at most this often. */
const PAINT_DELAY_MS = 150
/** Rendered text the stored text does not hold: buttons (code headers) and KaTeX's output (math is painted twice). */
const NOT_MESSAGE_TEXT = "button, .katex"

/** The CSS Custom Highlight registry, null in a browser without it: hits are then not painted, jumping to one still works. */
function highlightRegistry(): HighlightRegistry | null {
  return typeof CSS !== "undefined" && "highlights" in CSS && typeof Highlight !== "undefined" ? CSS.highlights : null
}

/** Every hit inside `element`, in text order (a hit's index is its ordinal). Matched per text node, as markup splits phrases. */
// shortcut: markdown the store counts as raw text (link targets, a phrase split by formatting) shifts the later ordinals of that text, the jump then lands near it.
export function elementHits(element: Element, find: TextFinder): Range[] {
  const ranges: Range[] = []
  const walker = document.createTreeWalker(element, NodeFilter.SHOW_TEXT, {
    acceptNode: (node) => (node.parentElement?.closest(NOT_MESSAGE_TEXT) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT),
  })
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    for (const span of find(node.nodeValue ?? "")) {
      const range = document.createRange()
      range.setStart(node, span.start)
      range.setEnd(node, span.end)
      ranges.push(range)
    }
  }
  return ranges
}

/** The elements whose `attribute` is `value`, in document order (e.g. an answer's text blocks). */
export function taggedElements(container: Element, attribute: string, value: string): Element[] {
  return [...container.querySelectorAll(`[${attribute}]`)].filter((element) => element.getAttribute(attribute) === value)
}

/** The shown hit's range, else null (its text is not rendered the way the store counted it). */
export function currentRange(container: Element, highlight: TextHighlight): Range | null {
  const find = highlight.matcher(highlight.query)
  const { current } = highlight
  if (!find || !current) return null
  return taggedElements(container, highlight.attribute, current.value).flatMap((element) => elementHits(element, find))[current.ordinal] ?? null
}

/**
 * Paints every hit of `highlight` in the texts mounted in `container` and the shown one apart, with the CSS Custom
 * Highlight API (the DOM stays untouched), repainted as the texts change. Cleared with the highlight.
 */
export function useTextHighlights(container: RefObject<HTMLElement | null>, highlight: TextHighlight | null): void {
  const name = highlight?.name
  const query = highlight?.query
  const matcher = highlight?.matcher
  const attribute = highlight?.attribute
  const value = highlight?.current?.value
  const ordinal = highlight?.current?.ordinal
  useEffect(() => {
    const element = container.current
    const registry = highlightRegistry()
    const find = query && matcher ? matcher(query) : null
    if (!element || !registry || !find || name === undefined || attribute === undefined) return
    const currentName = `${name}-current`
    let timer: ReturnType<typeof setTimeout> | undefined
    const paint = () => {
      timer = undefined
      const all = new Highlight()
      for (const text of element.querySelectorAll(`[${attribute}]`)) for (const range of elementHits(text, find)) all.add(range)
      registry.set(name, all)
      const range = value === undefined || ordinal === undefined ? undefined : taggedElements(element, attribute, value).flatMap((text) => elementHits(text, find))[ordinal]
      if (range) registry.set(currentName, new Highlight(range))
      else registry.delete(currentName)
    }
    const schedule = () => { timer ??= setTimeout(paint, PAINT_DELAY_MS) }
    paint()
    const observer = new MutationObserver(schedule)
    observer.observe(element, { childList: true, subtree: true, characterData: true })
    return () => {
      observer.disconnect()
      clearTimeout(timer)
      registry.delete(name)
      registry.delete(currentName)
    }
  }, [container, name, query, matcher, attribute, value, ordinal])
}

/** Scrolls `root` so `range` sits in its middle. Instant: a smooth scroll would race the sentinel that loads an earlier page. */
export function scrollToRange(root: HTMLElement, range: Range): void {
  range.startContainer.parentElement?.scrollIntoView({ block: "center", behavior: "instant" })
  const rect = range.getBoundingClientRect()
  const view = root.getBoundingClientRect()
  root.scrollTo({ top: root.scrollTop + rect.top + rect.height / 2 - (view.top + view.height / 2), behavior: "instant" })
}
