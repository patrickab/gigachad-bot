"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { createPortal } from "react-dom"
import { motion, AnimatePresence } from "framer-motion"
import { X, Check, Maximize2, Minimize2, Undo2, Trash2, Pencil } from "lucide-react"
import { cn } from "@/lib/utils"
import { getStroke } from "perfect-freehand"
import { type StrokeData, getSvgPathFromStroke, renderStrokesToJpeg } from "@/lib/drawing"
import { uploadFile as apiUploadFile, chatFileUrl } from "@/lib/api"
import { activeThemeName, readToken } from "@/lib/palette"
import type { Attachment } from "@/lib/types"

const THIN_WIDTH = 3
const THICK_WIDTH = 8

const STROKE_OPTIONS = {
  smoothing: 0.5,
  streamline: 0.5,
  simulatePressure: true,
  last: true,
} as const

function detectTheme(): "dark" | "light" {
  return activeThemeName()
}

interface DrawingCanvasProps {
  chatId: string
  onConfirm: (attachment: Attachment) => void
  onClose: () => void
  slug?: string | null
}

export function DrawingCanvas({ chatId, onConfirm, onClose, slug = null }: DrawingCanvasProps) {
  const [strokes, setStrokes] = useState<StrokeData[]>([])
  const [currentPoints, setCurrentPoints] = useState<number[][]>([])
  const [isDrawing, setIsDrawing] = useState(false)
  const [strokeWidth, setStrokeWidth] = useState<"thin" | "thick">("thin")
  const [isMaximized, setIsMaximized] = useState(false)
  const [isErasing, setIsErasing] = useState(false)
  const [isExporting, setIsExporting] = useState(false)
  const [theme, setTheme] = useState(detectTheme)

  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const observer = new MutationObserver(() => {
      setTheme(detectTheme())
    })
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    })
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    function handler(e: KeyboardEvent) {
      if (e.key === "Escape") {
        e.preventDefault()
        onClose()
      }
      if ((e.ctrlKey || e.metaKey) && e.key === "z") {
        e.preventDefault()
        setStrokes((prev) => prev.slice(0, -1))
      }
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [onClose])

  const baseWidth = strokeWidth === "thin" ? THIN_WIDTH : THICK_WIDTH
  const color = readToken("ink")
  const bgColor = readToken("surfaceElevated")

  const getPointerPos = useCallback(
    (e: React.PointerEvent<SVGSVGElement>) => {
      const svg = svgRef.current
      if (!svg) return [0, 0, 0.5]
      const rect = svg.getBoundingClientRect()
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top
      const pressure = e.pressure > 0 ? e.pressure : 0.5
      return [x, y, pressure]
    },
    [],
  )

  const handlePointerDown = useCallback(
    (e: React.PointerEvent<SVGSVGElement>) => {
      if (e.button === 5) {
        setIsErasing(true)
        return
      }
      if (e.button !== 0) return
      ;(e.target as Element).setPointerCapture(e.pointerId)
      setIsDrawing(true)
      const pos = getPointerPos(e)
      setCurrentPoints([pos])
    },
    [getPointerPos],
  )

  const handlePointerMove = useCallback(
    (e: React.PointerEvent<SVGSVGElement>) => {
      if (isErasing) {
        const svg = svgRef.current
        if (!svg) return
        const rect = svg.getBoundingClientRect()
        const x = e.clientX - rect.left
        const y = e.clientY - rect.top
        const threshold = 15
        setStrokes((prev) =>
          prev.filter((stroke) => {
            for (const pt of stroke.points) {
              const dx = pt[0]! - x
              const dy = pt[1]! - y
              if (Math.sqrt(dx * dx + dy * dy) < threshold) return false
            }
            return true
          }),
        )
        return
      }
      if (!isDrawing) return
      const pos = getPointerPos(e)
      setCurrentPoints((prev) => [...prev, pos])
    },
    [isDrawing, isErasing, getPointerPos],
  )

  const handlePointerUp = useCallback(() => {
    if (isErasing) {
      setIsErasing(false)
      return
    }
    if (!isDrawing) return
    setIsDrawing(false)
    if (currentPoints.length < 2) {
      setCurrentPoints([])
      return
    }
    setStrokes((prev) => [
      ...prev,
      { points: currentPoints, color, width: baseWidth },
    ])
    setCurrentPoints([])
  }, [isDrawing, isErasing, currentPoints, color, baseWidth])

  const handleUndo = useCallback(() => {
    setStrokes((prev) => prev.slice(0, -1))
  }, [])

  const handleClear = useCallback(() => {
    setStrokes([])
    setCurrentPoints([])
  }, [])

  const handleConfirm = useCallback(async () => {
    if (strokes.length === 0) return
    setIsExporting(true)
    try {
      const svg = svgRef.current
      if (!svg) return
      const width = svg.clientWidth || 600
      const height = svg.clientHeight || 400

      const blob = await renderStrokesToJpeg(strokes, width, height, bgColor)
      const timestamp = Date.now()
      const file = new File([blob], `drawing-${timestamp}.jpg`, { type: "image/jpeg" })
      const att = await apiUploadFile(chatId, file, slug)
      onConfirm(att)
    } catch {
      // silently fail
    } finally {
      setIsExporting(false)
    }
  }, [strokes, chatId, slug, onConfirm])

  const currentStrokeOutline =
    currentPoints.length >= 2
      ? getSvgPathFromStroke(
          getStroke(currentPoints, { ...STROKE_OPTIONS, size: baseWidth, last: false }),
        )
      : ""

  return createPortal(
    <AnimatePresence>
      <motion.div
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
            className={cn(
              "flex flex-col overflow-hidden rounded-xl border border-divider-strong shadow-[var(--shadow-xl)] bg-surface-elevated",
              isMaximized
                ? "fixed inset-4 z-[101]"
                : "w-full max-w-2xl mx-4 h-[70vh]",
            )}
          >
            <div className="flex items-center justify-between px-4 py-2 border-b border-divider-strong shrink-0">
              <div className="flex items-center gap-2 text-xs font-medium text-ink-muted">
                <Pencil className="h-3.5 w-3.5" />
                Drawing
                {isErasing && (
                  <span className="text-ink-faint">(eraser active)</span>
                )}
              </div>
              <div className="flex items-center gap-1">
                <button
                  onClick={handleUndo}
                  disabled={strokes.length === 0}
                  className="rounded p-1.5 text-ink-subtle hover:text-ink hover:bg-hover disabled:opacity-30 transition-colors"
                  title="Undo (Ctrl+Z)"
                >
                  <Undo2 className="h-4 w-4" />
                </button>
                <button
                  onClick={handleClear}
                  disabled={strokes.length === 0}
                  className="rounded p-1.5 text-ink-subtle hover:text-ink hover:bg-hover disabled:opacity-30 transition-colors"
                  title="Clear"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
                <div className="w-px h-4 bg-divider-strong mx-1" />
                <button
                  onClick={() => setStrokeWidth((w) => (w === "thin" ? "thick" : "thin"))}
                  className={cn(
                    "rounded px-2 py-1 text-[10px] font-medium transition-colors",
                    strokeWidth === "thick"
                      ? "text-ink bg-surface"
                      : "text-ink-subtle hover:text-ink",
                  )}
                  title="Toggle stroke width"
                >
                  {strokeWidth === "thin" ? "Thin" : "Thick"}
                </button>
                <button
                  onClick={() => setIsMaximized((m) => !m)}
                  className="rounded p-1.5 text-ink-subtle hover:text-ink hover:bg-hover transition-colors"
                  title={isMaximized ? "Minimize" : "Maximize"}
                >
                  {isMaximized ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
                </button>
                <div className="w-px h-4 bg-divider-strong mx-1" />
                <button
                  onClick={onClose}
                  className="rounded p-1.5 text-ink-subtle hover:text-danger transition-colors"
                  title="Close (Esc)"
                >
                  <X className="h-4 w-4" />
                </button>
                <button
                  onClick={handleConfirm}
                  disabled={strokes.length === 0 || isExporting}
                  className={cn(
                    "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors border",
                    strokes.length > 0 && !isExporting
                      ? "text-ink bg-surface-elevated border-divider-strong hover:bg-surface-elevated"
                      : "text-ink-faint border-divider-strong",
                  )}
                  title="Confirm drawing"
                >
                  {isExporting ? (
                    <span className="h-3.5 w-3.5 shrink-0 animate-spin rounded-full border-2 border-ink-faint border-t-ink" />
                  ) : (
                    <Check className="h-3.5 w-3.5" />
                  )}
                  {isExporting ? "Saving…" : "Confirm"}
                </button>
              </div>
            </div>

          <div
            ref={containerRef}
            className="flex-1 min-h-0 relative"
          >
            <svg
              ref={svgRef}
              className="w-full h-full"
              style={{ touchAction: "none" }}
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerLeave={handlePointerUp}
            >
              <rect width="100%" height="100%" fill="var(--surface-elevated)" />
              {strokes.map((stroke, i) => {
                const outline = getStroke(stroke.points, {
                  ...STROKE_OPTIONS,
                  size: stroke.width,
                })
                const pathD = getSvgPathFromStroke(outline)
                if (!pathD) return null
                return <path key={i} d={pathD} fill={stroke.color} />
              })}
              {currentStrokeOutline && (
                <path d={currentStrokeOutline} fill={color} />
              )}
            </svg>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>,
    document.body,
  )
}