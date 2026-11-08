"use client"

import { useState } from "react"
import { Document, Page } from "react-pdf"
import "./pdfjsWorker"

/** Every page of a PDF stacked at `width` px, each at its own aspect ratio, without text or annotation layers. `onFirstPageAspect` gets page 1's height / width. */
export function PdfPages({ url, width, onFirstPageAspect }: { url: string; width: number; onFirstPageAspect?: (ratio: number) => void }) {
  const [pageCount, setPageCount] = useState(0)
  return (
    <Document file={url} loading={null} error={null} onLoadSuccess={({ numPages }) => setPageCount(numPages)}>
      <div className="flex flex-col gap-2">
        {Array.from({ length: pageCount }, (_, index) => (
          <Page
            key={index}
            pageNumber={index + 1}
            width={width}
            renderTextLayer={false}
            renderAnnotationLayer={false}
            loading={null}
            onLoadSuccess={index === 0 && onFirstPageAspect ? (page) => onFirstPageAspect(page.originalHeight / page.originalWidth) : undefined}
            className="overflow-hidden rounded"
          />
        ))}
      </div>
    </Document>
  )
}
