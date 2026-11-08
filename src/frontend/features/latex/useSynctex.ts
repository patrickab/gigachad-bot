"use client"

import { useState } from "react"
import type { PdfReveal } from "@/components/content/PdfViewer"
import type { PdfWord } from "@/components/content/pdf/pdfText"
import type { ReportNotice } from "@/features/workbench/useWorkbenchNotice"
import { useLatestRequest } from "@/hooks/useLatestRequest"
import { ApiError, findSlideEnd, findSynctexSource, findSynctexView } from "@/lib/backend/api"
import { errorMessage } from "@/lib/utils"

/** Notice sources: a lookup's success clears only the notice of its own kind. */
const FORWARD_SEARCH = "forward search"
const INVERSE_SEARCH = "inverse search"
const SLIDE_GAP = "slide gap"

/** Why a SyncTeX lookup found nothing: `miss` when SyncTeX knows no match (404) or the match lies in a TeX file outside the mounts, e.g. a theme's (403), else the failure's message or `fallback`. */
function synctexNotice(cause: unknown, miss: string, fallback: string): string {
  const missed = cause instanceof ApiError && (cause.status === 404 || cause.status === 403)
  return missed ? miss : errorMessage(cause, fallback)
}

/**
 * SyncTeX forward search into the PDF the LaTeX `root` built: `reveal(path, line)` points `target` at the PDF spot of 1-based `line` of the source file at `path` and resolves whether it did.
 *
 * - Every found spot gets a new `target.key`, so the same spot is revealed again
 * - Only the latest line is revealed, an answer to an earlier one is dropped
 * - A line that revealed nothing reports why, the next one that did clears it
 */
export function useForwardSearch(root: string | null, report: ReportNotice) {
  const [target, setTarget] = useState<PdfReveal | undefined>(undefined)
  const { run } = useLatestRequest()

  const reveal = async (path: string, line: number) => {
    if (root === null) return false
    return run(
      findSynctexView(root, path, line),
      (spot) => {
        report(FORWARD_SEARCH, null)
        setTarget((shown) => ({ page: spot.page, y: spot.y, words: spot.words, key: (shown?.key ?? 0) + 1 }))
      },
      (cause) => report(FORWARD_SEARCH, synctexNotice(cause, "No PDF spot for this line", "Could not find the PDF spot")),
    )
  }

  return { reveal, target }
}

/**
 * SyncTeX inverse search over the PDF the LaTeX `root` built: `reveal(page, x, y, word, onSource)` hands the source file, 1-based line and 0-based UTF-8 byte column of that spot to `onSource`.
 *
 * - `x`, `y` are PDF points from the top-left corner of 1-based `page`, `word` the double-clicked word there or the page's word nearest it (empty when none), which picks its own line of a beamer frame, and its clicked character the column
 * - Only the latest spot is revealed, an answer to an earlier one is dropped
 * - A spot that revealed nothing reports why, the next one that did clears it
 */
export function useInverseSearch(root: string | null, report: ReportNotice) {
  const { run } = useLatestRequest()

  const reveal = (page: number, x: number, y: number, word: PdfWord, onSource: (path: string, line: number, column: number) => void) => {
    if (root === null) return
    void run(
      findSynctexSource(root, page, x, y, word.text, word.offset, word.occurrence),
      (source) => {
        report(INVERSE_SEARCH, null)
        onSource(source.path, source.line, source.column)
      },
      (cause) => report(INVERSE_SEARCH, synctexNotice(cause, "No source line for this spot", "Could not find the source line")),
    )
  }

  return { reveal }
}

/**
 * The insertion point of a presentation the LaTeX `root` builds: `choose(page, placeCursor)` hands `placeCursor` the line after the slide on 1-based `page`, so "Add slide" inserts the next frame there.
 *
 * - The line is the slide's `\end{frame}`, for a theme's chapter slide its `\section` (and the subsections right below it)
 * - `marked` is the chosen page once the cursor is there, until `clear`, the next `choose` or another root
 * - Only the latest page is chosen, an answer to an earlier one is dropped
 * - A page that took no cursor reports why, the next one that did or `clear` clears it
 */
export function useSlideGap(root: string | null, report: ReportNotice) {
  const [chosen, setChosen] = useState<{ root: string; page: number } | null>(null)
  const { run, cancel } = useLatestRequest()

  const choose = (page: number, placeCursor: (path: string, line: number) => void) => {
    if (root === null) return
    void run(
      findSlideEnd(root, page),
      (end) => {
        report(SLIDE_GAP, null)
        placeCursor(end.path, end.line)
        setChosen({ root, page })
      },
      (cause) => {
        setChosen(null)
        report(SLIDE_GAP, synctexNotice(cause, "No slide to insert after", "Could not find the slide"))
      },
    )
  }

  const clear = () => {
    cancel()
    setChosen(null)
    report(SLIDE_GAP, null)
  }

  return { marked: chosen?.root === root ? chosen.page : null, choose, clear }
}
