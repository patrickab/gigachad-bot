"use client"

import { Copy, Trash2 } from "lucide-react"
import { inkColor, strokePath } from "@/features/editor/canvas/ink/canvasInk"
import type { Poly, SelBox } from "@/features/editor/canvas/selection/canvasSelection"
import type { StrokeData } from "@/lib/drawing"

interface CanvasSelectionOutlineProps {
  lasso: Poly
  bounds: SelBox
  strokes: StrokeData[]
  selectedIds: Set<string>
  /** Strokes of an in-flight drag; the committed layer hides them, so they are drawn here instead. */
  preview: StrokeData[] | null
  scale: number
  isDark: boolean
  onDragStart: (handle: "move" | "resize", clientX: number, clientY: number) => void
}

/**
 * Stroke selection in canvas space — highlighted outlines plus the freehand lasso fill, which is
 * what you drag to move them together. One handle at the shape's bottom-right scales it
 * diagonally; it is sized in `u` so it stays a constant, thumb-findable size at any zoom.
 */
export function CanvasSelectionOutline({ lasso, bounds: b, strokes, selectedIds, preview, scale, isDark, onDragStart }: CanvasSelectionOutlineProps) {
  const u = 1 / scale
  const hit = 13 * u  // invisible grab area — bigger than the dot you see
  const dot = 5 * u
  // Mid-drag the committed layer hides these, so they are drawn here from the
  // preview — one gesture frame re-renders the selection, not the whole canvas.
  const picked = preview ?? strokes.filter((s) => selectedIds.has(s.id))
  return (
    <>
      {picked.map((s) => {
        const d = strokePath(s)
        if (!d) return null
        return (
          <g key={`sel-${s.id}`}>
            {preview && <path d={d} fill={inkColor(s.color, isDark)} />}
            <path d={d} fill="none" stroke="var(--ink)" strokeWidth={1.5 * u} strokeDasharray={`${3 * u} ${3 * u}`} />
          </g>
        )
      })}
      <path
        d={`M ${lasso.map(([x, y]) => `${x},${y}`).join(" L ")} Z`}
        fill="var(--ink-faint)" fillOpacity={0.08}
        stroke="var(--ink-muted)" strokeWidth={u} strokeDasharray={`${4 * u} ${4 * u}`}
        style={{ cursor: "grab", pointerEvents: "all" }}
        onPointerDown={(e) => { e.stopPropagation(); onDragStart("move", e.clientX, e.clientY) }}
      />
      <g
        style={{ cursor: "nwse-resize" }}
        onPointerDown={(e) => { e.stopPropagation(); onDragStart("resize", e.clientX, e.clientY) }}
      >
        <rect x={b.x + b.w - hit / 2} y={b.y + b.h - hit / 2} width={hit} height={hit} fill="transparent" />
        <rect
          x={b.x + b.w - dot / 2} y={b.y + b.h - dot / 2} width={dot} height={dot} rx={1.5 * u}
          fill="var(--paper)" stroke="var(--ink-muted)" strokeWidth={u}
          style={{ pointerEvents: "none" }}
        />
      </g>
    </>
  )
}

/** Duplicate / delete buttons, anchored just above the lasso's bounds (in screen space). */
export function CanvasSelectionActions({ bounds, scale, offset, onCopy, onDelete }: {
  bounds: SelBox
  scale: number
  offset: { x: number; y: number }
  onCopy: () => void
  onDelete: () => void
}) {
  return (
    <div
      className="absolute z-10 flex items-center gap-1 -translate-y-full"
      style={{ left: bounds.x * scale + offset.x, top: bounds.y * scale + offset.y - 6 }}
    >
      <button aria-label="Copy selection"
        onPointerDown={(e) => e.stopPropagation()}
        onClick={(e) => { e.stopPropagation(); onCopy() }}
        className="rounded-full p-1 bg-paper border border-divider-strong text-ink-faint hover:text-ink shadow-[var(--shadow-lg)] transition-colors"
      >
        <Copy className="h-3 w-3" />
      </button>
      <button aria-label="Delete selection"
        onPointerDown={(e) => e.stopPropagation()}
        onClick={(e) => { e.stopPropagation(); onDelete() }}
        className="rounded-full p-1 bg-paper border border-divider-strong text-ink-faint hover:text-danger shadow-[var(--shadow-lg)] transition-colors"
      >
        <Trash2 className="h-3 w-3" />
      </button>
    </div>
  )
}

/** In-progress lasso — the pen-button or toolbar drag traces the selection outline freehand (canvas space). */
export function CanvasLassoTrace({ points, scale }: { points: Poly; scale: number }) {
  if (points.length < 2) return null
  return (
    <path
      d={`M ${points.map(([x, y]) => `${x},${y}`).join(" L ")}`}
      fill="none"
      stroke="var(--ink-muted)" strokeWidth={1.5 / scale} strokeDasharray={`${4 / scale} ${4 / scale}`}
      style={{ pointerEvents: "none" }}
    />
  )
}

/** Screenshot drag rectangle between two canvas points, above live attachments so it stays visible over a PDF (screen space). */
export function CanvasScreenshotBox({ from, to, scale, offset }: { from: [number, number]; to: [number, number]; scale: number; offset: { x: number; y: number } }) {
  return (
    <div
      data-testid="screenshot-selection"
      className="absolute z-10 pointer-events-none border-[1.5px] border-dashed border-ink bg-ink/5"
      style={{
        left: Math.min(from[0], to[0]) * scale + offset.x,
        top: Math.min(from[1], to[1]) * scale + offset.y,
        width: Math.abs(to[0] - from[0]) * scale,
        height: Math.abs(to[1] - from[1]) * scale,
      }}
    />
  )
}
