"use client"

import { useEffect, useState } from "react"
import { LazyPdfViewer } from "@/components/content/LazyPdfViewer"
import type { PdfPageBox } from "@/components/content/PdfViewer"
import { TOP_BAR_PX } from "@/features/workbench/WorkbenchSplit"

/** Longest time the progress bar waits for a new PDF to swap in, e.g. one that failed to load never does. */
const SETTLE_TIMEOUT_MS = 2000

/**
 * The built PDF. A new `url` for the same `documentKey` reloads in place and keeps zoom and scroll position.
 *
 * - The pages stay sharp and the old build stays on screen until the new one's pages have drawn
 * - A thin progress bar shows while a build runs and until the new PDF has swapped in
 * - Without a PDF it shows the build progress or `emptyMessage`
 * - Pages lay out in `pageBox`, lined up with the editor
 */
export function LatexPdfPane({ url, documentKey, building, emptyMessage, pageBox }: {
  url: string | null
  documentKey: string | null
  building: boolean
  emptyMessage: string | null
  pageBox: PdfPageBox
}) {
  // The url whose pages are on screen, null before the first PDF drew
  const [drawnUrl, setDrawnUrl] = useState<string | null>(null)
  const settling = drawnUrl !== null && url !== null && url !== drawnUrl
  useEffect(() => {
    if (!settling || url === null) return
    const timer = setTimeout(() => setDrawnUrl(url), SETTLE_TIMEOUT_MS)
    return () => clearTimeout(timer)
  }, [settling, url])

  return (
    <div className="relative h-full overflow-hidden">
      {(building || settling) && (
        <div role="progressbar" aria-label="Building" className="absolute inset-x-0 top-11 z-10 h-0.5 overflow-hidden">
          <div className="indeterminate-bar h-full rounded-full bg-ink-faint" />
        </div>
      )}
      {url && documentKey ? (
        <div className="h-full">
          <LazyPdfViewer url={url} documentKey={documentKey} topInset={TOP_BAR_PX} pageBox={pageBox} onPageRendered={setDrawnUrl} />
        </div>
      ) : (
        <div className="flex h-full items-center justify-center text-xs text-ink-subtle">{building ? "Building…" : emptyMessage}</div>
      )}
    </div>
  )
}
