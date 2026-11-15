"use client"

import { useEffect, useRef, useState } from "react"
import { Page } from "react-pdf"
import type { PdfDocument } from "@/components/content/pdf/pdfLayout"
import { useElementSize } from "@/hooks/useElementSize"
import { cn } from "@/lib/utils"
import { useDeckEmbeds } from "./slideEmbeds"

/** 4:3, the aspect of a slide whose size is unknown or failed to load. */
export const FALLBACK_ASPECT = 3 / 4

/** Page `pageNumber` scaled to fit the box and centered in it, in the PDF's true colours (no dark-theme inversion). The last drawn page stays up until the next one has rendered, so changing slides never flashes. With `pdf`, every link of the deck whose address carries `embed=1` is loaded up front in a live frame, which covers the link and takes the keyboard while its slide shows. */
export function SlideView({ pageNumber, aspects, pdf }: { pageNumber: number; aspects: number[]; pdf?: PdfDocument }) {
  const boxRef = useRef<HTMLDivElement>(null)
  const { width, height } = useElementSize(boxRef)
  const [drawn, setDrawn] = useState<number | null>(null)
  const embeds = useDeckEmbeds(pdf)
  const framesRef = useRef(new Map<string, HTMLIFrameElement>())
  const frameKey = (page: number, url: string) => `${page} ${url}`
  const shownFrame = () => {
    const shown = embeds.find((embed) => embed.page === pageNumber)
    return shown ? framesRef.current.get(frameKey(shown.page, shown.url)) : undefined
  }
  const pageAspect = aspects[pageNumber - 1] ?? FALLBACK_ASPECT
  const shownWidth = Math.floor(Math.min(width, height / pageAspect))

  const tellFramesWhetherShown = () => {
    for (const { page, url } of embeds) framesRef.current.get(frameKey(page, url))?.contentWindow?.postMessage({ type: "deck-shown", shown: page === pageNumber }, "*")
  }

  // A hidden applet pauses. The keyboard goes to the shown applet, and back to the deck on a slide without one
  useEffect(() => {
    tellFramesWhetherShown()
    const frame = shownFrame()
    if (frame) frame.focus()
    else if (document.activeElement instanceof HTMLIFrameElement) document.activeElement.blur()
  })
  const layers = drawn === null || drawn === pageNumber ? [pageNumber] : [drawn, pageNumber]

  return <div ref={boxRef} className="pdf-true-colors relative h-full w-full overflow-hidden">
    {layers.map((layer) => {
      const pageWidth = Math.floor(Math.min(width, height / (aspects[layer - 1] ?? FALLBACK_ASPECT)))
      const waiting = layer !== drawn && drawn !== null
      return <div key={layer} aria-hidden={waiting || undefined} className={cn("absolute inset-0 flex items-center justify-center", waiting && "invisible")}>
        {pageWidth > 0 && <Page
          pageNumber={layer}
          width={pageWidth}
          renderTextLayer={false}
          renderAnnotationLayer={false}
          loading={null}
          onRenderSuccess={() => setDrawn(layer)}
        />}
      </div>
    })}
    {embeds.length > 0 && shownWidth > 0 && <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
      <div className="relative" style={{ width: shownWidth, height: shownWidth * pageAspect }}>
        {embeds.map(({ page, url, left, top, width, height }) => <iframe
          key={frameKey(page, url)}
          ref={(frame) => {
            if (frame) framesRef.current.set(frameKey(page, url), frame)
            else framesRef.current.delete(frameKey(page, url))
          }}
          src={url}
          title="Slide applet"
          tabIndex={page === pageNumber ? 0 : -1}
          onLoad={() => {
            tellFramesWhetherShown()
            shownFrame()?.focus()
          }}
          className={cn("absolute border-0", page === pageNumber ? "pointer-events-auto" : "invisible")}
          style={{ left: `${100 * left}%`, top: `${100 * top}%`, width: `${100 * width}%`, height: `${100 * height}%` }}
        />)}
      </div>
    </div>}
  </div>
}
