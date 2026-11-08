"use client"

import { useRef, useState } from "react"
import { Page } from "react-pdf"
import { useElementSize } from "@/hooks/useElementSize"
import { cn } from "@/lib/utils"

/** 4:3, the aspect of a slide whose size is unknown or failed to load. */
export const FALLBACK_ASPECT = 3 / 4

/** Page `pageNumber` scaled to fit the box and centered in it, in the PDF's true colours (no dark-theme inversion). The last drawn page stays up until the next one has rendered, so changing slides never flashes. */
export function SlideView({ pageNumber, aspects }: { pageNumber: number; aspects: number[] }) {
  const boxRef = useRef<HTMLDivElement>(null)
  const { width, height } = useElementSize(boxRef)
  const [drawn, setDrawn] = useState<number | null>(null)
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
  </div>
}
