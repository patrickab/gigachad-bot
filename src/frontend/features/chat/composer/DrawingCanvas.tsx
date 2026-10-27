"use client"

import { useCallback, useRef, useState } from "react"
import { createPortal } from "react-dom"
import { motion } from "framer-motion"
import { X, Check, Maximize2, Minimize2, Undo2, Trash2, Pencil } from "lucide-react"
import { cn, isModKey } from "@/lib/utils"
import { type StrokeData, STROKE_OPTIONS as BASE_STROKE_OPTIONS, penPressure, strokeOutlinePath } from "@/lib/drawing"
import { renderStrokesToJpeg } from "@/lib/canvasExport"
import { uploadFile as apiUploadFile } from "@/lib/backend/api"
import { ModalBackdrop } from "@/components/ui/ModalBackdrop"
import { useEventListener } from "@/hooks/useEventListener"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import { useThemeName } from "@/hooks/useThemeName"
import { readToken } from "@/lib/palette"
import type { Attachment } from "@/lib/domain/types"

const THIN_WIDTH = 3
const THICK_WIDTH = 8
/** Erase radius in px around the pointer. */
const ERASE_RADIUS = 15

// Live pen input carries no real pressure, so simulate it here. The shared base
// in lib/drawing renders already-captured strokes and leaves it off.
const STROKE_OPTIONS = { ...BASE_STROKE_OPTIONS, simulatePressure: true } as const

/** What the pointer is doing on the canvas: drawing a stroke with its points so far, or erasing (pen eraser tip). */
type PointerState = { kind: "draw"; points: number[][] } | { kind: "erase" }

/** `strokes` without those that pass within `radius` of (`x`, `y`). */
function eraseNear(strokes: StrokeData[], x: number, y: number, radius: number): StrokeData[] {
  return strokes.filter((stroke) => stroke.points.every((pt) => {
    const dx = pt[0]! - x
    const dy = pt[1]! - y
    return Math.sqrt(dx * dx + dy * dy) >= radius
  }))
}

interface DrawingCanvasProps {
  chatId: string
  onConfirm: (attachment: Attachment) => void
  onClose: () => void
  slug?: string | null
}

export function DrawingCanvas({ chatId, onConfirm, onClose, slug = null }: DrawingCanvasProps) {
  const [strokes, setStrokes] = useState<StrokeData[]>([])
  const [pointer, setPointer] = useState<PointerState | null>(null)
  const [strokeWidth, setStrokeWidth] = useState<"thin" | "thick">("thin")
  const [isMaximized, setIsMaximized] = useState(false)
  const [isExporting, setIsExporting] = useState(false)
  useThemeName() // re-read the palette tokens below when the theme flips

  const svgRef = useRef<SVGSVGElement>(null)

  const handleUndo = useCallback(() => {
    setStrokes((prev) => prev.slice(0, -1))
  }, [])

  useEscapeLayer(onClose)

  useEventListener("document", "keydown", (e) => {
    if (isModKey(e) && e.key === "z") {
      e.preventDefault()
      handleUndo()
    }
  })

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
      return [x, y, penPressure(e)]
    },
    [],
  )

  const handlePointerDown = useCallback(
    (e: React.PointerEvent<SVGSVGElement>) => {
      // Ignore touch (palm/hand), only pen and mouse draw
      if (e.pointerType === "touch") return
      // Button 5 is the pen eraser tip
      if (e.button === 5) {
        setPointer({ kind: "erase" })
        return
      }
      if (e.button !== 0) return
      ;(e.target as Element).setPointerCapture(e.pointerId)
      setPointer({ kind: "draw", points: [getPointerPos(e)] })
    },
    [getPointerPos],
  )

  const handlePointerMove = useCallback(
    (e: React.PointerEvent<SVGSVGElement>) => {
      if (e.pointerType === "touch" || !pointer || !svgRef.current) return
      const pos = getPointerPos(e)
      if (pointer.kind === "erase") setStrokes((prev) => eraseNear(prev, pos[0]!, pos[1]!, ERASE_RADIUS))
      else setPointer((current) => current?.kind === "draw" ? { kind: "draw", points: [...current.points, pos] } : current)
    },
    [pointer, getPointerPos],
  )

  const handlePointerUp = useCallback((e: React.PointerEvent<SVGSVGElement>) => {
    if (e.pointerType === "touch" || !pointer) return
    setPointer(null)
    if (pointer.kind === "draw" && pointer.points.length >= 2) {
      setStrokes((prev) => [
        ...prev,
        { id: crypto.randomUUID(), points: pointer.points, color, width: baseWidth },
      ])
    }
  }, [pointer, color, baseWidth])

  const handleClear = useCallback(() => {
    setStrokes([])
    setPointer((current) => current?.kind === "draw" ? { kind: "draw", points: [] } : current)
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
      // No error UI. The dialog stays open with strokes intact so the user can retry.
    } finally {
      setIsExporting(false)
    }
  }, [strokes, chatId, slug, onConfirm, bgColor])

  const currentPoints = pointer?.kind === "draw" ? pointer.points : []
  const currentStrokeOutline = currentPoints.length >= 2
    ? strokeOutlinePath(currentPoints, baseWidth, { ...STROKE_OPTIONS, last: false })
    : ""

  // No `onClose` on the backdrop: a click outside must not discard the strokes.
  return createPortal(
    <ModalBackdrop className="z-[100] backdrop-blur-xl">
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
            {pointer?.kind === "erase" && (
              <span className="text-ink-faint">(eraser active)</span>
            )}
          </div>
          <div className="flex items-center gap-1">
            <button aria-label="Undo stroke"
              onClick={handleUndo}
              disabled={strokes.length === 0}
              className="rounded p-1.5 text-ink-subtle hover:text-ink hover:bg-hover disabled:opacity-30 transition-colors"
            >
              <Undo2 className="h-4 w-4" />
            </button>
            <button aria-label="Clear drawing"
              onClick={handleClear}
              disabled={strokes.length === 0}
              className="rounded p-1.5 text-ink-subtle hover:text-ink hover:bg-hover disabled:opacity-30 transition-colors"
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
            >
              {strokeWidth === "thin" ? "Thin" : "Thick"}
            </button>
            <button
              aria-label={isMaximized ? "Restore drawing window" : "Maximize drawing window"}
              onClick={() => setIsMaximized((m) => !m)}
              className="rounded p-1.5 text-ink-subtle hover:text-ink hover:bg-hover transition-colors"
            >
              {isMaximized ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
            </button>
            <div className="w-px h-4 bg-divider-strong mx-1" />
            <button aria-label="Close drawing"
              onClick={onClose}
              className="rounded p-1.5 text-ink-subtle hover:text-danger transition-colors"
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

        <div className="flex-1 min-h-0 relative">
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
              const pathD = strokeOutlinePath(stroke.points, stroke.width, STROKE_OPTIONS)
              if (!pathD) return null
              return <path key={i} d={pathD} fill={stroke.color} />
            })}
            {currentStrokeOutline && (
              <path d={currentStrokeOutline} fill={color} />
            )}
          </svg>
        </div>
      </motion.div>
    </ModalBackdrop>,
    document.body,
  )
}
