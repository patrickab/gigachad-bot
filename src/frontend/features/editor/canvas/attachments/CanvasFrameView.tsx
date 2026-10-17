"use client"

import { fileViewerRawUrl } from "@/lib/backend/urls"
import type { CanvasFrame } from "@/lib/domain/canvasDocument"

interface CanvasFrameViewProps {
  frame: CanvasFrame
  /** Height / width of the rendered frame. */
  aspect: number
  /** Effective canvas zoom; all chrome is sized in one screen px (`u = 1 / scale`) so it stays constant at any zoom. */
  scale: number
  onDragStart: (mode: "move" | "resize", clientX: number, clientY: number) => void
  onRemove: () => void
  /** Reports an image frame's measured height / width. */
  onAspect: (aspect: number) => void
}

/** A page or image frame: pointer-transparent content with a top drag band, resize corner and delete button. */
export function CanvasFrameView({ frame: f, aspect, scale, onDragStart, onRemove, onAspect }: CanvasFrameViewProps) {
  const u = 1 / scale // one screen pixel in canvas units
  const w = f.width
  const h = f.width * aspect
  const rx = 6 * u
  const band = 16 * u           // top drag-band height
  const gripW = 26 * u          // grip handle pill
  const delC = { x: f.x + w - 11 * u, y: f.y + 11 * u } // delete center (inside top-right)
  const arm = 2.75 * u
  return (
    <g className="group/frame">
      {/* content — clipped to rounded corners, pointer-transparent so you can draw over it */}
      <clipPath id={`frame-clip-${f.id}`}>
        <rect x={f.x} y={f.y} width={w} height={h} rx={rx} ry={rx} />
      </clipPath>
      <g clipPath={`url(#frame-clip-${f.id})`} style={{ pointerEvents: "none" }}>
        {f.kind === "page" ? (
          <rect x={f.x} y={f.y} width={w} height={h} fill="var(--surface-elevated)" />
        ) : (
          <image
            href={fileViewerRawUrl(f.path!)}
            x={f.x} y={f.y} width={w} height={h}
            preserveAspectRatio="none"
            onLoad={(e) => {
              const img = e.currentTarget as unknown as SVGImageElement & { naturalWidth?: number; naturalHeight?: number }
              if (img.naturalWidth && img.naturalHeight) onAspect(img.naturalHeight / img.naturalWidth)
              else {
                const probe = new Image()
                probe.onload = () => { if (probe.naturalWidth) onAspect(probe.naturalHeight / probe.naturalWidth) }
                probe.src = fileViewerRawUrl(f.path!)
              }
            }}
          />
        )}
      </g>
      {/* rounded border */}
      <rect x={f.x} y={f.y} width={w} height={h} rx={rx} ry={rx} fill="none" stroke="var(--divider-strong)" strokeWidth={u} style={{ pointerEvents: "none" }} />

      {/* top drag-band hit area (move) */}
      <rect
        x={f.x} y={f.y} width={w} height={band}
        fill="transparent" style={{ cursor: "grab" }}
        onPointerDown={(e) => { e.stopPropagation(); onDragStart("move", e.clientX, e.clientY) }}
      />
      {/* grip handle — the minimal drag affordance, dim until hover */}
      <rect
        x={f.x + w / 2 - gripW / 2} y={f.y + 2.5 * u} width={gripW} height={3.5 * u} rx={1.75 * u}
        fill="var(--ink-faint)"
        className="opacity-50 group-hover/frame:opacity-90 transition-opacity"
        style={{ pointerEvents: "none" }}
      />

      {/* resize handle (bottom-right) — subtle corner ticks */}
      <g
        style={{ cursor: "nwse-resize" }}
        onPointerDown={(e) => { e.stopPropagation(); onDragStart("resize", e.clientX, e.clientY) }}
      >
        <rect x={f.x + w - 16 * u} y={f.y + h - 16 * u} width={16 * u} height={16 * u} fill="transparent" />
        <path
          d={`M ${f.x + w - 4.5 * u},${f.y + h - 1.5 * u} L ${f.x + w - 1.5 * u},${f.y + h - 4.5 * u} M ${f.x + w - 9 * u},${f.y + h - 1.5 * u} L ${f.x + w - 1.5 * u},${f.y + h - 9 * u}`}
          stroke="var(--ink-faint)" strokeWidth={1.25 * u} strokeLinecap="round" fill="none"
          className="opacity-50 group-hover/frame:opacity-90 transition-opacity"
          style={{ pointerEvents: "none" }}
        />
      </g>

      {/* delete — always visible, grey → red on hover, no background (matches CanvasAttachment) */}
      <g
        className="text-ink-faint hover:text-danger transition-colors"
        style={{ cursor: "pointer" }}
        onPointerDown={(e) => e.stopPropagation()}
        onClick={(e) => { e.stopPropagation(); onRemove() }}
      >
        <circle cx={delC.x} cy={delC.y} r={8 * u} fill="transparent" />
        <path
          d={`M ${delC.x - arm},${delC.y - arm} L ${delC.x + arm},${delC.y + arm} M ${delC.x - arm},${delC.y + arm} L ${delC.x + arm},${delC.y - arm}`}
          stroke="currentColor" strokeWidth={1.5 * u} strokeLinecap="round" fill="none"
        />
      </g>
    </g>
  )
}
