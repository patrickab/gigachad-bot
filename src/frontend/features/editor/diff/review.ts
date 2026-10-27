import { useCallback, useEffect, useState } from "react"
import type { PatchLine } from "@/lib/diff"
import { readStorage, writeStorage } from "@/lib/storage"

export type ReviewSide = "old" | "new"

/** Original line identity and quoted diff context. */
export interface NoteAnchor {
  path: string
  side: ReviewSide
  line: number
  lineText: string
  /** Up to two diff lines per side, retaining their prefixes. */
  quote: string[]
  /** Target offset in quote, absent in drafts saved before context matching. */
  quoteIndex?: number
  selectedText?: string
}

export interface ReviewNote extends NoteAnchor {
  id: string
  comment: string
}

const QUOTE_CONTEXT = 2

const PREFIX: Partial<Record<PatchLine["kind"], string>> = { added: "+", removed: "-", context: " " }

/** Anchor at `lines[index]`, quoting neighbours within the same hunk. */
export function anchorAt(path: string, lines: PatchLine[], index: number, side: ReviewSide): NoteAnchor | null {
  const target = lines[index]
  const line = target && (side === "old" ? target.oldNumber : target.newNumber)
  if (!target || !line) return null
  let start = index
  while (start > index - QUOTE_CONTEXT && lines[start - 1] && lines[start - 1]!.kind !== "hunk") start--
  let end = index
  while (end < index + QUOTE_CONTEXT && lines[end + 1] && lines[end + 1]!.kind !== "hunk") end++
  const quoted = lines.slice(start, end + 1).filter((line) => PREFIX[line.kind] !== undefined)
  const quote = quoted.map((line) => PREFIX[line.kind] + line.text)
  return { path, side, line, lineText: target.text, quote, quoteIndex: quoted.indexOf(target) }
}

/** Resolve by line text and adjacent same-side context, leaving ambiguous matches outdated. */
export function locateNote(anchor: NoteAnchor, lines: PatchLine[]): number | null {
  const quoteMatches = anchor.quote.flatMap((text, index) => text.slice(1) === anchor.lineText && (text[0] === " " || text[0] === (anchor.side === "old" ? "-" : "+")) ? [index] : [])
  const quoteIndex = anchor.quoteIndex ?? (quoteMatches.length === 1 ? quoteMatches[0] : undefined)
  if (quoteIndex === undefined) return null
  const sideQuote = anchor.quote.flatMap((text, index) => text[0] === " " || text[0] === (anchor.side === "old" ? "-" : "+") ? [{ text: text.slice(1), index }] : [])
  const target = sideQuote.findIndex((line) => line.index === quoteIndex)
  if (target < 0) return null
  const before = sideQuote.slice(0, target).map((line) => line.text).reverse()
  const after = sideQuote.slice(target + 1).map((line) => line.text)
  let best: number | null = null
  let bestScore = -1
  let ambiguous = false
  for (let index = 0; index < lines.length; index++) {
    const line = lines[index]!
    const number = anchor.side === "old" ? line.oldNumber : line.newNumber
    if (number === null || line.text !== anchor.lineText) continue
    let score = 0
    for (const [direction, expected] of [[-1, before], [1, after]] as const) {
      let cursor = index + direction
      for (const text of expected) {
        while (lines[cursor] && lines[cursor]!.kind !== "hunk" && (anchor.side === "old" ? lines[cursor]!.oldNumber : lines[cursor]!.newNumber) === null) cursor += direction
        if (!lines[cursor] || lines[cursor]!.kind === "hunk" || lines[cursor]!.text !== text) break
        score++
        cursor += direction
      }
    }
    if (score === 0 && before.length + after.length > 0) continue
    if (score > bestScore) {
      best = index
      bestScore = score
      ambiguous = false
    } else if (score === bestScore) ambiguous = true
  }
  return ambiguous ? null : best
}

/** Format current locations and context, explicitly marking unresolved original locations. */
export function formatReview(notes: ReviewNote[], patches: Record<string, PatchLine[] | null | undefined>): string {
  const resolved = notes.map((note) => {
    const lines = patches[note.path]
    const index = lines ? locateNote(note, lines) : null
    const current = lines && index !== null ? anchorAt(note.path, lines, index, note.side) : null
    return { ...note, ...current, outdated: current === null }
  }).sort((a, b) => a.path.localeCompare(b.path) || a.line - b.line)
  const sections = resolved.map((note, index) => [
    `${index + 1}. \`${note.path}:${note.line}\`${note.side === "old" ? " (old side)" : ""}${note.outdated ? " (outdated, original location)" : ""}`,
    "```diff", ...note.quote, "```",
    ...(note.selectedText ? ["Selected code:", "```", note.selectedText, "```"] : []),
    note.comment,
  ].join("\n"))
  return ["Address each review note using its quoted diff context.", ...sections].join("\n\n")
}

const storageKey = (root: string) => `diff-review:${root}`

function loadNotes(root: string): ReviewNote[] {
  try {
    const stored: unknown = JSON.parse(readStorage(storageKey(root)) ?? "[]")
    return Array.isArray(stored) ? stored as ReviewNote[] : []
  } catch {
    return []
  }
}

/** Discard pending notes for a removed worktree. */
export function discardReviewNotes(root: string): void {
  localStorage.removeItem(storageKey(root))
}

/** Persist pending notes per worktree across remounts and reloads. */
export function useReviewNotes(root: string) {
  const [state, setState] = useState(() => ({ root, notes: loadNotes(root) }))
  if (state.root !== root) setState({ root, notes: loadNotes(root) })

  useEffect(() => {
    if (state.notes.length > 0) writeStorage(storageKey(state.root), JSON.stringify(state.notes))
    else discardReviewNotes(state.root)
  }, [state])

  const update = useCallback((change: (notes: ReviewNote[]) => ReviewNote[]) => setState((previous) => ({ root: previous.root, notes: change(previous.notes) })), [])
  return { notes: state.root === root ? state.notes : [], update }
}
