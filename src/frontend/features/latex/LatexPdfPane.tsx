"use client"

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react"
import { LazyPdfViewer } from "@/components/content/LazyPdfViewer"
import type { PdfPageBox, PdfReveal } from "@/components/content/PdfViewer"
import { PAGE_GAP } from "@/components/content/pdf/pdfLayout"
import type { PdfWord } from "@/components/content/pdf/pdfText"
import { TOP_BAR_PX } from "@/features/workbench/WorkbenchSplit"

/** Longest time the progress bar waits for a new PDF to swap in, e.g. one that failed to load never does. */
const SETTLE_TIMEOUT_MS = 2000

/** The clickable gap below 1-based `page`, in the page's bottom margin so it shifts no layout: a thin line on hover and while `marked`. */
function SlideGapButton({ page, marked, onClick }: { page: number; marked: boolean; onClick: (page: number) => void }) {
  return (
    <button
      type="button"
      tabIndex={-1}
      aria-label={`Gap after page ${page}`}
      aria-pressed={marked}
      className="group absolute inset-x-0 top-full z-10 flex cursor-pointer items-center focus:outline-none"
      style={{ height: PAGE_GAP }}
      onClick={() => onClick(page)}
      // A double-click here is not one on the page above
      onDoubleClick={(event) => event.stopPropagation()}
    >
      <span className={`h-0.5 w-full rounded-full transition-opacity ${marked ? "bg-ink opacity-100" : "bg-ink-subtle opacity-0 group-hover:opacity-100"}`} />
    </button>
  )
}

/**
 * The built PDF. A new `url` for the same `documentKey` reloads in place and keeps zoom and scroll position.
 *
 * - The pages stay sharp and the old build stays on screen until the new one's pages have drawn
 * - A thin progress bar shows while a build runs and until the new PDF has swapped in
 * - Without a PDF it shows the build progress or `emptyMessage`
 * - Pages lay out in `pageBox`, lined up with the editor
 * - `onPageDoubleClick` gets the double-clicked page, the spot on it in PDF points from its top-left corner, and the word selected there or the page's word nearest it (empty when none) with the clicked character's offset in it and its occurrence on its rendered line
 * - `reveal` scrolls to a spot and marks it, a new `key` repeats it
 * - `slideGap` makes the gaps between pages clickable (`onChoose` gets the page above), keeps a line in the gap below its `marked` page, Escape in the pane clears it (`onClear`)
 */
export function LatexPdfPane({ url, documentKey, building, emptyMessage, pageBox, onPageDoubleClick, reveal, slideGap }: {
  url: string | null
  documentKey: string | null
  building: boolean
  emptyMessage: string | null
  pageBox: PdfPageBox
  onPageDoubleClick?: (page: number, x: number, y: number, word: PdfWord) => void
  reveal?: PdfReveal
  slideGap?: { marked: number | null; onChoose: (page: number) => void; onClear: () => void }
}) {
  // The url whose pages are on screen, null before the first PDF drew
  const [drawnUrl, setDrawnUrl] = useState<string | null>(null)
  const settling = drawnUrl !== null && url !== null && url !== drawnUrl
  useEffect(() => {
    if (!settling || url === null) return
    const timer = setTimeout(() => setDrawnUrl(url), SETTLE_TIMEOUT_MS)
    return () => clearTimeout(timer)
  }, [settling, url])

  // The latest `onChoose`, which the parent hands in anew on every render
  const chooseRef = useRef(slideGap?.onChoose)
  useLayoutEffect(() => {
    chooseRef.current = slideGap?.onChoose
  })
  const choose = useCallback((page: number) => chooseRef.current?.(page), [])
  const gaps = slideGap !== undefined
  const marked = slideGap?.marked ?? null
  // Stable while the mark stays, so pages do not redraw their overlays on every parent render
  const renderGap = useCallback((page: number, _pageWidth: number, pageCount: number) => (
    page < pageCount && <SlideGapButton page={page} marked={marked === page} onClick={choose} />
  ), [marked, choose])

  return (
    <div
      className="relative h-full overflow-hidden"
      onKeyDown={(event) => {
        if (event.key === "Escape" && slideGap?.marked != null) slideGap.onClear()
      }}
    >
      {(building || settling) && (
        <div role="progressbar" aria-label="Building" className="absolute inset-x-0 top-11 z-10 h-0.5 overflow-hidden">
          <div className="indeterminate-bar h-full rounded-full bg-ink-faint" />
        </div>
      )}
      {url && documentKey ? (
        <div className="h-full">
          <LazyPdfViewer
            url={url}
            documentKey={documentKey}
            topInset={TOP_BAR_PX}
            pageBox={pageBox}
            onPageRendered={setDrawnUrl}
            onPageDoubleClick={onPageDoubleClick}
            reveal={reveal}
            renderPageOverlay={gaps ? renderGap : undefined}
          />
        </div>
      ) : (
        <div className="flex h-full items-center justify-center text-xs text-ink-subtle">{building ? "Building…" : emptyMessage}</div>
      )}
    </div>
  )
}
