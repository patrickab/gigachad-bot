"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { Document, Page, pdfjs } from "react-pdf"
import "react-pdf/dist/esm/Page/AnnotationLayer.css"
import "react-pdf/dist/esm/Page/TextLayer.css"

pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`

const RENDER_WIDTH = 900
const PAGE_GAP = 8

export function PdfViewer({ url }: { url: string }) {
  const [numPages, setNumPages] = useState<number | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [scale, setScale] = useState<number | null>(null)

  useEffect(() => {
    const el = containerRef.current
    if (!el) return

    const updateScale = () => {
      const w = el.clientWidth
      if (w > 0) setScale(Math.min(1, (w - 16) / RENDER_WIDTH))
    }

    updateScale()

    const ro = new ResizeObserver(updateScale)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const onDocumentLoadSuccess = useCallback(({ numPages }: { numPages: number }) => {
    setNumPages(numPages)
  }, [])

  return (
    <div ref={containerRef} className="w-full h-full min-h-[60vh] overflow-auto">
      <Document file={url} onLoadSuccess={onDocumentLoadSuccess}>
        {scale != null && Array.from({ length: numPages ?? 0 }, (_, i) => (
          <div key={i} style={{ marginBottom: PAGE_GAP }}>
            <div style={{ zoom: scale }}>
              <Page pageNumber={i + 1} width={RENDER_WIDTH} />
            </div>
          </div>
        ))}
      </Document>
    </div>
  )
}