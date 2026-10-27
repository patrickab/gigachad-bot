"use client"

import { useState, type ReactNode } from "react"
import { FileType, Maximize2, Minimize2, SquarePen, X } from "lucide-react"
import { attachmentName, attachmentTraits } from "@/lib/domain/canvasAttachments"
import { CANVAS_ASPECT, type AttachmentPatch, type CanvasAttachment, type NestedCanvasAttachment } from "@/lib/domain/canvasDocument"
import type { Point } from "@/lib/domain/edgeRouting"
import { cn } from "@/lib/utils"
import { ArchitectureWindow } from "@/features/editor/architecture/ArchitectureWindow"
import { CanvasResizeHandle } from "@/features/editor/canvas/CanvasResizeHandle"
import { AnnotatedPdfViewer } from "@/features/editor/pdf/AnnotatedPdfViewer"
import { CanvasDocumentAttachment } from "./CanvasDocumentAttachment"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"

const PDF_LAYOUT_CAP = 916 // PdfViewer caps rendering at 900px + its 8px side gaps — wider layout gains no resolution
const PLOT_MIN_LAYOUT_WIDTH = 960 // plots lay out at least this wide and scale down, so legends/titles never collide
const TITLE_BAR_PX = 31

interface CanvasAttachmentWindowProps {
  attachment: CanvasAttachment
  /** Display name the canvas's project gives the attachment's file, if any. */
  shownName?: string
  /** Effective canvas zoom, the zoom it last settled at, and pan: the window is positioned HTML over the SVG. */
  scale: number
  settledScale: number
  offset: Point
  /** Height / width of a PDF page, until measured a fallback. */
  pdfAspect: number
  /** Blown up over the whole surface; fullscreen only restyles this wrapper so the editor inside keeps its state. */
  full: boolean
  /** Screenshot selection drags over the window, so it must not swallow the pointer. */
  inert: boolean
  onDragStart: (mode: "move" | "resize", clientX: number, clientY: number) => void
  onToggleFullscreen: () => void
  onRemove: () => void
  onPatch: (patch: AttachmentPatch) => void
  onRetitle: (title: string) => void
  onPdfAspect: (aspect: number) => void
  /** The canvas pen or highlighter, a stylus on a PDF window writes on the PDF with it. */
  pdfPen: { color: string; width: number; highlight: boolean }
  /** Body of a canvas window. A prop, not an import, so editor and window stay out of an import cycle. */
  renderNestedCanvas: (attachment: NestedCanvasAttachment, name: string) => ReactNode
}

/** A live, scrollable window over the canvas for a PDF, document, architecture diagram or nested canvas. Never baked into export. */
export function CanvasAttachmentWindow({
  attachment: att, shownName, scale, settledScale, offset, pdfAspect, full, inert,
  onDragStart, onToggleFullscreen, onRemove, onPatch, onRetitle, onPdfAspect, pdfPen, renderNestedCanvas,
}: CanvasAttachmentWindowProps) {
  const [editingTitle, setEditingTitle] = useState(false)
  useEscapeLayer(() => setEditingTitle(false), editingTitle)
  const pdf = att.kind === "pdf"
  const fixedSize = attachmentTraits[att.kind].fixedHeight
  const aspect = pdf ? pdfAspect : CANVAS_ASPECT
  const name = attachmentName(att, shownName)
  // PDFs rasterize through a settled layout size. Interactive viewport frames
  // use their actual screen size so pointer coordinates stay exact.
  const screenX = att.x * scale + offset.x
  const screenY = att.y * scale + offset.y
  const contentHeight = att.height ?? att.width * CANVAS_ASPECT
  const screenW = att.width * scale
  const screenH = contentHeight * scale
  const layoutW = pdf ? Math.min(att.width * settledScale, PDF_LAYOUT_CAP) : screenW

  // One body per kind; each sees its own fields.
  const body = () => {
    switch (att.kind) {
      case "architecture-graph":
        return <ArchitectureWindow path={att.path} maximized={full} hostScale={full ? 1 : scale} edgeStyle={att.edgeStyle} onEdgeStyleChange={(edgeStyle) => onPatch({ edgeStyle })} onNavigate={(path) => onPatch({ path })} />
      case "document":
        return (
          <CanvasDocumentAttachment
            path={att.path}
            hostScale={scale}
            plotFrame={full ? null : { width: Math.max(att.width * settledScale, PLOT_MIN_LAYOUT_WIDTH), scale: screenW / Math.max(att.width * settledScale, PLOT_MIN_LAYOUT_WIDTH) }}
            collapsed={!full && scale !== settledScale}
          />
        )
      case "canvas":
        return renderNestedCanvas(att, name)
      case "pdf":
        return (
          <AnnotatedPdfViewer
            path={att.path}
            pen={pdfPen}
            clipToPages
            onPageAspect={onPdfAspect}
            initialPage={att.page}
            onPageChange={(page) => onPatch({ page })}
          />
        )
    }
  }

  const commitTitle = (title: string) => {
    setEditingTitle(false)
    onRetitle(title)
  }

  return (
    <div
      className={cn(
        "absolute flex flex-col border border-divider-strong rounded-lg overflow-hidden bg-paper shadow-[var(--shadow-lg)]",
        full && "z-20",
        inert && "pointer-events-none",
      )}
      style={full
        ? { inset: 0, transform: "none" }
        : fixedSize
          ? { left: screenX, top: screenY, width: screenW, height: screenH, transform: "none" }
          : { left: screenX, top: screenY, width: layoutW, transform: `scale(${screenW / layoutW})`, transformOrigin: "top left" }}
    >
      <div
        className={cn(
          "flex items-center gap-1.5 px-2 py-1 border-b border-divider/50 select-none shrink-0",
          !full && "cursor-grab active:cursor-grabbing",
        )}
        onPointerDown={(e) => { e.stopPropagation(); if (!full) onDragStart("move", e.clientX, e.clientY) }}
      >
        {att.kind === "canvas" ? <SquarePen className="h-3 w-3 text-ink-faint shrink-0" /> : <FileType className="h-3 w-3 text-ink-faint shrink-0" />}
        {editingTitle ? (
          <input
            autoFocus
            defaultValue={name}
            className="flex-1 min-w-0 bg-transparent text-[10px] font-medium text-ink outline-none select-text"
            onPointerDown={(e) => e.stopPropagation()}
            onBlur={(e) => commitTitle(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") commitTitle(e.currentTarget.value) }}
          />
        ) : (
          <span
            className="flex-1 min-w-0 truncate text-[10px] font-medium text-ink-muted"
            onDoubleClick={(e) => { e.stopPropagation(); setEditingTitle(true) }}
          >
            {name}
          </span>
        )}
        <button
          aria-label={full ? `Restore ${name}` : `Expand ${name}`}
          onClick={(e) => { e.stopPropagation(); onToggleFullscreen() }}
          className="rounded p-0.5 text-ink-faint hover:text-ink transition-colors shrink-0"
        >
          {full ? <Minimize2 className="h-3 w-3" /> : <Maximize2 className="h-3 w-3" />}
        </button>
        <button aria-label="Remove attachment"
          onClick={(e) => { e.stopPropagation(); onRemove() }}
          className="rounded p-0.5 text-ink-faint hover:text-danger transition-colors shrink-0"
        >
          <X className="h-3 w-3" />
        </button>
      </div>
      {/* Body height: fixed-size windows subtract the title bar, PDFs follow their measured aspect. */}
      <div
        data-canvas-attachment
        className={cn(full && "flex-1 min-h-0")}
        style={{ height: full ? undefined : fixedSize ? screenH - TITLE_BAR_PX : layoutW * aspect }}
        onPointerDown={(e) => e.stopPropagation()}
      >
        {body()}
      </div>
      {!full && <CanvasResizeHandle className="w-4 h-4" onDragStart={onDragStart} />}
    </div>
  )
}
