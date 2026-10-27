import { anchorAt, type NoteAnchor, type ReviewSide } from "./review"
import type { FilePatch } from "./useRootDiff"

export type SelectedComment = { anchor: NoteAnchor; bounds: DOMRect }

/** Capture code only, excluding gutters and comments, within one file and split-view side. */
export function selectedComment(container: HTMLElement, patches: Record<string, FilePatch>): SelectedComment | null {
  const selection = window.getSelection()
  if (!selection || selection.isCollapsed || selection.rangeCount !== 1) return null
  const range = selection.getRangeAt(0)
  const startElement = range.startContainer.nodeType === Node.ELEMENT_NODE ? range.startContainer as Element : range.startContainer.parentElement
  const endElement = range.endContainer.nodeType === Node.ELEMENT_NODE ? range.endContainer as Element : range.endContainer.parentElement
  const start = startElement?.closest<HTMLElement>("[data-diff-code]")
  const end = endElement?.closest<HTMLElement>("[data-diff-code]")
  if (!start || !end || !container.contains(start) || !container.contains(end) || start.dataset.path !== end.dataset.path || start.dataset.column !== end.dataset.column) return null
  const path = start.dataset.path!
  const lines = patches[path]?.lines
  if (!lines) return null
  const text: string[] = []
  let lastCode: HTMLElement | null = null
  let bounds: DOMRect | null = null
  for (const code of start.closest("section")!.querySelectorAll<HTMLElement>("[data-diff-code]")) {
    if (code.dataset.column !== start.dataset.column || !range.intersectsNode(code)) continue
    const part = document.createRange()
    part.selectNodeContents(code)
    if (range.compareBoundaryPoints(Range.START_TO_START, part) > 0) part.setStart(range.startContainer, range.startOffset)
    if (range.compareBoundaryPoints(Range.END_TO_END, part) < 0) part.setEnd(range.endContainer, range.endOffset)
    const selected = part.toString()
    const blankInside = code.textContent === "" && range.compareBoundaryPoints(Range.START_TO_START, part) < 0 && range.compareBoundaryPoints(Range.END_TO_END, part) > 0
    if (selected || blankInside) {
      text.push(selected)
      lastCode = code
      const rectangle = part.getBoundingClientRect()
      bounds = rectangle.height > 0 ? rectangle : code.getBoundingClientRect()
    }
  }
  const selectedText = text.join("\n")
  if (!selectedText.trim() || !lastCode || !bounds) return null
  const anchor = anchorAt(path, lines, Number(lastCode.dataset.index), lastCode.dataset.side as ReviewSide)
  if (!anchor) return null
  return { anchor: { ...anchor, selectedText }, bounds }
}
