"use client"

import { useEffect, useState } from "react"
import { LazyPdfViewer } from "@/components/content/LazyPdfViewer"
import type { PdfPageBox } from "@/components/content/PdfViewer"
import { cn } from "@/lib/utils"
import { TOP_BAR_PX } from "@/features/workbench/WorkbenchSplit"

/** Longest time the pane stays blurred for a new PDF's pages to draw. */
const SETTLE_TIMEOUT_MS = 2000

/**
 * The built PDF. A new `url` for the same `documentKey` reloads in place and keeps zoom and scroll position.
 *
 * - While the agent edits (`editing`) the pages blur and dim, behind a thin progress bar and a quiet label
 * - While a build runs the bar shows alone
 * - A new PDF stays blurred until its first page has drawn, so the pane never flashes blank
 * - Without a PDF it shows the build progress or `emptyMessage`
 * - Pages lay out in `pageBox`, lined up with the editor
 */
export function LatexPdfPane({ url, documentKey, building, editing, emptyMessage, pageBox }: {
  url: string | null
  documentKey: string | null
  building: boolean
  editing: boolean
  emptyMessage: string | null
  pageBox: PdfPageBox
}) {
  // The url whose pages last drew, null before the first PDF drew
  const [drawnUrl, setDrawnUrl] = useState<string | null>(null)
  const settling = drawnUrl !== null && url !== null && url !== drawnUrl
  useEffect(() => {
    if (!settling || url === null) return
    const timer = setTimeout(() => setDrawnUrl(url), SETTLE_TIMEOUT_MS)
    return () => clearTimeout(timer)
  }, [settling, url])

  return (
    <div className="relative h-full overflow-hidden">
      {(building || editing) && (
        <div role="progressbar" aria-label={editing ? "omp is editing" : "Building"} className="absolute inset-x-0 top-11 z-10 h-0.5 overflow-hidden">
          <div className="indeterminate-bar h-full rounded-full bg-ink-faint" />
        </div>
      )}
      {editing && (
        <p role="status" className="absolute left-1/2 top-14 z-10 -translate-x-1/2 rounded-full bg-paper/80 px-2.5 py-1 text-[11px] text-ink-subtle">omp is editing</p>
      )}
      {url && documentKey ? (
        <div className={cn("h-full transition-[filter,opacity] duration-200", (editing || settling) && "opacity-70 blur-[2px]")}>
          <LazyPdfViewer url={url} documentKey={documentKey} topInset={TOP_BAR_PX} pageBox={pageBox} onPageRendered={() => setDrawnUrl(url)} />
        </div>
      ) : (
        <div className="flex h-full items-center justify-center text-xs text-ink-subtle">{building ? "Building…" : emptyMessage}</div>
      )}
    </div>
  )
}
