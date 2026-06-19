"use client"

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react"
import { createPortal } from "react-dom"
import { Document, Page, pdfjs } from "react-pdf"
import { Maximize2, Minimize2, Minus, Plus } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"
import "react-pdf/dist/esm/Page/AnnotationLayer.css"
import "react-pdf/dist/esm/Page/TextLayer.css"

pdfjs.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs"

const RENDER_WIDTH = 900
const PAGE_GAP = 1
const MIN_ZOOM = 1
const MAX_ZOOM = 2.5
const ZOOM_STEP = 0.25
const ZOOM_PRESETS = [1, 1.25, 1.5, 1.75, 2, 2.5]

interface ScrollAnchor {
  mouseX: number
  mouseY: number
  scrollLeft: number
  scrollTop: number
  ratio: number
}

function PdfViewerInner({
  url,
  isFullscreen,
  onToggleFullscreen,
}: {
  url: string
  isFullscreen: boolean
  onToggleFullscreen: () => void
}) {
  const [numPages, setNumPages] = useState<number | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const [containerWidth, setContainerWidth] = useState<number | null>(null)
  const [zoom, setZoom] = useState(1)
  const zoomRef = useRef(1)
  const anchorRef = useRef<ScrollAnchor | null>(null)
  const hideTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [controlsVisible, setControlsVisible] = useState(false)
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = scrollRef.current
    if (!el) return

    const update = () => {
      const w = el.clientWidth
      if (w > 0) setContainerWidth(w)
    }

    update()

    const ro = new ResizeObserver(update)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  useEffect(() => {
    if (!dropdownOpen) return
    function handleClick(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setDropdownOpen(false)
      }
    }
    document.addEventListener("mousedown", handleClick)
    return () => document.removeEventListener("mousedown", handleClick)
  }, [dropdownOpen])

  const showControls = useCallback(() => {
    if (hideTimerRef.current) clearTimeout(hideTimerRef.current)
    setControlsVisible(true)
    hideTimerRef.current = setTimeout(() => {
      setControlsVisible(false)
      setDropdownOpen(false)
    }, 2500)
  }, [])

  useEffect(() => {
    const el = scrollRef.current
    if (!el) return

    const onWheel = (e: WheelEvent) => {
      if (!e.ctrlKey && !e.metaKey) return
      e.preventDefault()

      const prevZoom = zoomRef.current
      const factor = e.deltaY < 0 ? 1.08 : 1 / 1.08
      const nextZoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, prevZoom * factor))
      if (nextZoom === prevZoom) return

      const rect = el.getBoundingClientRect()
      anchorRef.current = {
        mouseX: e.clientX - rect.left,
        mouseY: e.clientY - rect.top,
        scrollLeft: el.scrollLeft,
        scrollTop: el.scrollTop,
        ratio: nextZoom / prevZoom,
      }

      zoomRef.current = nextZoom
      setZoom(nextZoom)
      showControls()
    }

    el.addEventListener("wheel", onWheel, { passive: false })
    return () => el.removeEventListener("wheel", onWheel)
  }, [showControls])

  useLayoutEffect(() => {
    const anchor = anchorRef.current
    if (!anchor) return
    anchorRef.current = null

    const el = scrollRef.current
    if (!el) return

    el.scrollLeft = anchor.scrollLeft * anchor.ratio + anchor.mouseX * (anchor.ratio - 1)
    el.scrollTop = anchor.scrollTop * anchor.ratio + anchor.mouseY * (anchor.ratio - 1)
  }, [zoom])

  const adjustZoom = useCallback((nextZoom: number) => {
    const el = scrollRef.current
    if (!el) return

    const prevZoom = zoomRef.current
    const clamped = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, nextZoom))
    if (clamped === prevZoom) return

    const rect = el.getBoundingClientRect()
    anchorRef.current = {
      mouseX: rect.width / 2,
      mouseY: rect.height / 2,
      scrollLeft: el.scrollLeft,
      scrollTop: el.scrollTop,
      ratio: clamped / prevZoom,
    }

    zoomRef.current = clamped
    setZoom(clamped)
    showControls()
  }, [showControls])

  const onDocumentLoadSuccess = useCallback(({ numPages }: { numPages: number }) => {
    setNumPages(numPages)
  }, [])

  const baseScale = containerWidth != null ? Math.min(1, (containerWidth - 16) / RENDER_WIDTH) : null
  const renderWidth = baseScale != null ? Math.round(RENDER_WIDTH * baseScale) : RENDER_WIDTH

  const zoomLabel = `${Math.round(zoom * 100)}%`

  return (
    <div
      className="relative w-full h-full min-h-0 flex flex-col"
      onMouseEnter={showControls}
      onMouseMove={showControls}
    >
      <div
        ref={dropdownRef}
        className={`absolute top-3 right-3 z-10 flex items-center gap-1 rounded-lg bg-surface-elevated/90 backdrop-blur-sm border border-divider/60 shadow-[var(--shadow-md)] transition-opacity duration-300 pointer-events-auto ${controlsVisible ? "opacity-100" : "opacity-0 pointer-events-none"}`}
      >
        <button
          onClick={() => adjustZoom(zoomRef.current - ZOOM_STEP)}
          disabled={zoom <= MIN_ZOOM}
          className="rounded-l-lg p-1.5 text-ink-subtle hover:text-ink hover:bg-hover disabled:text-ink-faint disabled:pointer-events-none transition-colors"
        >
          <Minus className="h-3.5 w-3.5" strokeWidth={2} />
        </button>
        <div className="relative">
          <button
            onClick={() => setDropdownOpen((o) => !o)}
            className="px-1.5 py-1.5 text-[10px] font-medium tabular-nums text-ink-muted hover:text-ink hover:bg-hover transition-colors select-none"
          >
            {zoomLabel}
          </button>
          <AnimatePresence>
            {dropdownOpen && (
              <motion.div
                initial={{ opacity: 0, y: -4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                transition={{ duration: 0.12 }}
                className="absolute top-full left-1/2 -translate-x-1/2 mt-1 py-1 rounded-lg bg-surface-elevated border border-divider shadow-[var(--shadow-lg)] z-20"
              >
                {ZOOM_PRESETS.map((preset) => (
                  <button
                    key={preset}
                    onClick={() => {
                      adjustZoom(preset)
                      setDropdownOpen(false)
                    }}
                    className={`block w-full px-4 py-1 text-[10px] font-medium tabular-nums text-left hover:bg-hover transition-colors ${
                      Math.abs(zoom - preset) < 0.01 ? "text-ink" : "text-ink-muted"
                    }`}
                  >
                    {Math.round(preset * 100)}%
                  </button>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
        <button
          onClick={() => adjustZoom(zoomRef.current + ZOOM_STEP)}
          disabled={zoom >= MAX_ZOOM}
          className="p-1.5 text-ink-subtle hover:text-ink hover:bg-hover disabled:text-ink-faint disabled:pointer-events-none transition-colors"
        >
          <Plus className="h-3.5 w-3.5" strokeWidth={2} />
        </button>
        <div className="w-px h-4 bg-divider" />
        <button
          onClick={onToggleFullscreen}
          className="rounded-r-lg p-1.5 text-ink-subtle hover:text-ink hover:bg-hover transition-colors"
          title={isFullscreen ? "Exit fullscreen" : "Fullscreen"}
        >
          {isFullscreen ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
        </button>
      </div>
      <div ref={scrollRef} className="flex-1 min-h-0 overflow-auto">
        <Document file={url} onLoadSuccess={onDocumentLoadSuccess}>
          {baseScale != null && numPages != null && (
            <div style={{ width: renderWidth * zoom }}>
              <div style={{ transform: `scale(${zoom})`, transformOrigin: "0 0" }}>
                {Array.from({ length: numPages }, (_, i) => (
                  <div key={i} style={{ marginBottom: PAGE_GAP }}>
                    <Page pageNumber={i + 1} width={renderWidth} />
                  </div>
                ))}
              </div>
            </div>
          )}
        </Document>
      </div>
    </div>
  )
}

export function PdfViewer({ url }: { url: string }) {
  const [isFullscreen, setIsFullscreen] = useState(false)

  useEffect(() => {
    if (!isFullscreen) return
    function handler(e: KeyboardEvent) {
      if (e.key === "Escape") {
        e.preventDefault()
        setIsFullscreen(false)
      }
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [isFullscreen])

  const toggleFullscreen = useCallback(() => setIsFullscreen((f) => !f), [])

  if (!isFullscreen) {
    return <PdfViewerInner url={url} isFullscreen={false} onToggleFullscreen={toggleFullscreen} />
  }

  return createPortal(
    <AnimatePresence>
      <motion.div
        key="pdf-fullscreen"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.15 }}
        className="fixed inset-0 z-[100] flex items-center justify-center bg-backdrop backdrop-blur-xl"
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.96, y: 8 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.96, y: 8 }}
          transition={{ duration: 0.2, ease: "easeOut" }}
          className="flex flex-col w-full h-full bg-paper rounded-lg overflow-hidden"
        >
          <PdfViewerInner url={url} isFullscreen={true} onToggleFullscreen={toggleFullscreen} />
        </motion.div>
      </motion.div>
    </AnimatePresence>,
    document.body,
  )
}