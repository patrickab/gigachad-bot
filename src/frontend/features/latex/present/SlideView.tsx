"use client"

import { useRef, useState } from "react"
import { Page } from "react-pdf"
import type { PdfDocument } from "@/components/content/pdf/pdfLayout"
import { useElementSize } from "@/hooks/useElementSize"
import { cn } from "@/lib/utils"
import { usePageEmbeds } from "./slideEmbeds"

/** 4:3, the aspect of a slide whose size is unknown or failed to load. */
export const FALLBACK_ASPECT = 3 / 4

/** Page `pageNumber` scaled to fit the box and centered in it, in the PDF's true colours (no dark-theme inversion). The last drawn page stays up until the next one has rendered, so changing slides never flashes. With `pdf`, every link of the page whose address carries `embed=1` is covered by that address in a live frame, which takes the keyboard. */
export function SlideView({ pageNumber, aspects, pdf }: { pageNumber: number; aspects: number[]; pdf?: PdfDocument }) {
  const boxRef = useRef<HTMLDivElement>(null)
  const { width, height } = useElementSize(boxRef)
  const [drawn, setDrawn] = useState<number | null>(null)
  const embeds = usePageEmbeds(pdf, pageNumber)
  const layers = drawn === null || drawn === pageNumber ? [pageNumber] : [drawn, pageNumber]

  return <div ref={boxRef} className="pdf-true-colors relative h-full w-full overflow-hidden">
    {layers.map((layer) => {
      const pageWidth = Math.floor(Math.min(width, height / (aspects[layer - 1] ?? FALLBACK_ASPECT)))
      const waiting = layer !== drawn && drawn !== null
      return <div key={layer} aria-hidden={waiting || undefined} className={cn("absolute inset-0 flex items-center justify-center", waiting && "invisible")}>
        {pageWidth > 0 && <div className="relative">
          <Page
            pageNumber={layer}
            width={pageWidth}
            renderTextLayer={false}
            renderAnnotationLayer={false}
            loading={null}
            onRenderSuccess={() => setDrawn(layer)}
          />
          {layer === pageNumber && embeds.map(({ url, left, top, width, height }) => <iframe
            key={url}
            src={url}
            title="Slide applet"
            onLoad={(event) => event.currentTarget.focus()}
            className="absolute border-0"
            style={{ left: `${100 * left}%`, top: `${100 * top}%`, width: `${100 * width}%`, height: `${100 * height}%` }}
          />)}
        </div>}
      </div>
    })}
  </div>
}
