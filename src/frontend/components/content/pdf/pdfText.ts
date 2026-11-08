import type { PdfPage } from "./pdfLayout"

/** Longest selection taken as the double-clicked word, a longer one is not a word. */
const MAX_WORD_LENGTH = 64
/** A letter or digit, which a word holds and a blank or punctuation does not. */
const WORD_CHARACTER = /[\p{L}\p{N}]/u
/** A word of page text: a run of letters and digits. */
const WORD = /[\p{L}\p{N}]+/gu
/** Share of a line's height two text items of it lie apart by at least where a space separates them, more than a kern. */
const SPACE_SHARE = 0.1

/** The word at a double-clicked spot: its `text` (empty when none), the 0-based `offset` of the clicked character in it and its 0-based `occurrence` among the equal words of its rendered line. */
export type PdfWord = { text: string; offset: number; occurrence: number }

const NO_WORD: PdfWord = { text: "", offset: 0, occurrence: 0 }

/** The word a double-click selected and the offset in it of the clicked character, null when the browser measured none. */
export type SelectedWord = { text: string; offset: number | null }

/** A text item of a page, placed in PDF points from the page's top-left corner. */
type PageText = { str: string; left: number; baseline: number; width: number; height: number }

/** The text items of *pdfPage* placed in PDF points from its top-left corner, none when its text fails to load. */
async function pageTexts(pdfPage: PdfPage): Promise<PageText[]> {
  const viewport = pdfPage.getViewport({ scale: 1 })
  const items = await pdfPage.getTextContent().then(({ items }) => items, () => [])
  const texts: PageText[] = []
  for (const item of items) {
    if (!("str" in item)) continue
    const [left, baseline] = viewport.convertToViewportPoint(item.transform[4], item.transform[5])
    texts.push({ str: item.str, left, baseline, width: item.width, height: item.height })
  }
  return texts
}

/** PDF points down from the top of *pdfPage* to the middle of its first text item holding one of *words*, tried in order, else *y* on the page, else 0. */
export async function revealedY(pdfPage: PdfPage, y: number, words: readonly string[] | undefined): Promise<number> {
  if (words?.length) {
    const texts = await pageTexts(pdfPage)
    for (const word of words) {
      const text = texts.find((entry) => entry.str.includes(word))
      // The mark is centred on the spot, so the line's middle puts it over the line instead of half a line above it
      if (text) return Math.max(0, text.baseline - text.height / 2)
    }
  }
  return y >= 0 && y <= pdfPage.getViewport({ scale: 1 }).height ? y : 0
}

/** The word a double-click at `clientX` in `slot` selected in its text layer, empty for a blank, punctuation, too long or outside selection. */
export function doubleClickedWord(slot: Element, clientX: number): SelectedWord {
  const selection = window.getSelection()
  if (!selection?.anchorNode || !slot.contains(selection.anchorNode) || selection.rangeCount === 0) return { text: "", offset: null }
  const selected = selection.toString()
  const text = selected.trim()
  if (text.length > MAX_WORD_LENGTH || !WORD_CHARACTER.test(text)) return { text: "", offset: null }
  const character = clickedCharacter(selection.getRangeAt(0), clientX)
  return { text, offset: character === null ? null : character - (selected.length - selected.trimStart().length) }
}

/** The index, among the characters of `range`, of the one whose box lies nearest `clientX`, null when the browser measures none. */
function clickedCharacter(range: Range, clientX: number): number | null {
  const glyph = document.createRange()
  if (typeof glyph.getBoundingClientRect !== "function") return null
  const root = range.commonAncestorContainer
  const nodes: Node[] = root.nodeType === Node.TEXT_NODE ? [root] : []
  for (const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT); walker.nextNode();) nodes.push(walker.currentNode)
  let index = 0
  let nearest: { index: number; gap: number } | null = null
  for (const node of nodes) {
    if (!range.intersectsNode(node)) continue
    const end = node === range.endContainer ? range.endOffset : node.textContent?.length ?? 0
    for (let offset = node === range.startContainer ? range.startOffset : 0; offset < end; offset++, index++) {
      glyph.setStart(node, offset)
      glyph.setEnd(node, offset + 1)
      const box = glyph.getBoundingClientRect()
      const gap = Math.max(box.left - clientX, 0, clientX - box.right)
      if (box.width > 0 && (!nearest || gap < nearest.gap)) nearest = { index, gap }
    }
  }
  return nearest?.index ?? null
}

/** How far `spot` lies outside the run of `length` characters at `start`, 0 inside it. */
function gapTo(spot: number, start: number, length: number): number {
  return Math.max(start - spot, 0, spot - start - length)
}

/** Where `word` stands in `text` as a whole word, not inside a longer run of letters and digits. */
function wholeWordStarts(text: string, word: string): number[] {
  const starts: number[] = []
  for (let start = text.indexOf(word); start >= 0; start = text.indexOf(word, start + 1)) {
    if (!WORD_CHARACTER.test(text[start - 1] ?? "") && !WORD_CHARACTER.test(text[start + word.length] ?? "")) starts.push(start)
  }
  return starts
}

/** The text of the rendered line of `nearest` among `texts`, its items on that baseline left to right, and the spot `x` as a character position in it, taking glyphs as equally wide. */
function renderedLine(texts: readonly PageText[], nearest: PageText, x: number): { line: string; spot: number } {
  const items = texts.filter((text) => Math.abs(text.baseline - nearest.baseline) <= nearest.height / 2).sort((a, b) => a.left - b.left)
  let line = ""
  let spot = 0
  let previous: PageText | null = null
  for (const item of items) {
    const apart = previous !== null && item.left - (previous.left + previous.width) > previous.height * SPACE_SHARE
    if (apart && !/\s$/.test(line) && !/^\s/.test(item.str)) line += " "
    if (item === nearest) spot = line.length + ((x - item.left) / item.width) * item.str.length
    line += item.str
    previous = item
  }
  return { line, spot }
}

/** The word at the spot `x`, `y` (PDF points from the top-left corner of *pdfPage*), `selected` or else the page's word nearest the spot (empty when no text is within a line height), with its occurrence among the equal whole words of the rendered line of the text item nearest the spot and its offset, the selected one or else the character under the spot taking glyphs as equally wide, in code points. */
export async function clickedWord(pdfPage: PdfPage, x: number, y: number, selected: SelectedWord): Promise<PdfWord> {
  const texts = (await pageTexts(pdfPage)).filter((text) => text.width > 0 && text.height > 0 && WORD_CHARACTER.test(text.str))
  let nearest: { text: PageText; distance: number } | null = null
  for (const text of texts) {
    const dx = Math.max(text.left - x, 0, x - text.left - text.width)
    const dy = Math.max(text.baseline - text.height - y, 0, y - text.baseline)
    if (dx > text.height || dy > text.height) continue
    const distance = Math.hypot(dx, dy)
    if (!nearest || distance < nearest.distance) nearest = { text, distance }
  }
  const { line, spot } = nearest ? renderedLine(texts, nearest.text, x) : { line: "", spot: 0 }
  let word = selected.text
  if (!word) {
    let gap = Infinity
    for (const match of line.matchAll(WORD)) {
      const wordGap = gapTo(spot, match.index, match[0].length)
      if (wordGap < gap) [word, gap] = [match[0], wordGap]
    }
    if (word.length > MAX_WORD_LENGTH) return NO_WORD
  }
  if (!word) return NO_WORD
  const starts = wholeWordStarts(line, word)
  let occurrence = 0
  starts.forEach((start, index) => {
    if (gapTo(spot, start, word.length) < gapTo(spot, starts[occurrence], word.length)) occurrence = index
  })
  const offset = selected.offset ?? (starts.length > 0 ? Math.floor(spot - starts[occurrence]) : 0)
  const clamped = Math.min(Math.max(offset, 0), word.length - 1)
  return { text: word, offset: Array.from(word.slice(0, clamped)).length, occurrence }
}
