"use client"

import { useCallback, useRef } from "react"
import { emptyCanvasDoc, type CanvasDocument, type NestedCanvasAttachment } from "@/lib/domain/canvasDocument"
import { LoadFailure, LoadingPlaceholder } from "@/components/ui/DocumentLoadState"
import { inkColor } from "@/features/editor/canvas/ink/canvasInk"
import { useInkSurface } from "@/features/editor/canvas/ink/useInkSurface"
import { useCanvasCamera } from "@/features/editor/canvas/camera/useCanvasCamera"
import { useCanvasDrag } from "@/features/editor/canvas/selection/useCanvasDrag"
import { useCanvasSelection } from "@/features/editor/canvas/selection/useCanvasSelection"
import { useRubberBand, type Band } from "@/features/editor/canvas/selection/useRubberBand"
import { useCanvasAttachments } from "@/features/editor/canvas/attachments/useCanvasAttachments"
import { useCanvasInsert } from "@/features/editor/canvas/attachments/useCanvasInsert"
import { useCanvasScreenshot } from "@/features/editor/canvas/attachments/useCanvasScreenshot"
import { useCanvasHistory } from "@/features/editor/canvas/useCanvasHistory"
import { useCollaborativeCanvas } from "@/features/editor/canvas/useCollaborativeCanvas"
import { useTabActive } from "@/contexts/TabContext"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import { useUndoRedoKeys } from "@/hooks/useUndoRedoKeys"
import { cn } from "@/lib/utils"
import { CanvasAddMenu } from "@/features/editor/canvas/attachments/CanvasAddMenu"
import { CanvasAttachmentWindow } from "@/features/editor/canvas/attachments/CanvasAttachmentWindow"
import { CanvasDotGrid } from "@/features/editor/canvas/camera/CanvasDotGrid"
import { CanvasFrameView } from "@/features/editor/canvas/attachments/CanvasFrameView"
import { CanvasLassoTrace, CanvasScreenshotBox, CanvasSelectionActions, CanvasSelectionOutline } from "@/features/editor/canvas/selection/CanvasSelectionOverlay"
import { CanvasStrokeLayer } from "@/features/editor/canvas/ink/CanvasStrokeLayer"
import { CanvasContextMenu } from "./CanvasContextMenu"
import { CanvasTextNote } from "./CanvasTextNote"
import { CanvasToolbar, useInkSettings } from "./CanvasToolbar"
import { useCanvasPointer } from "./useCanvasPointer"
import { useCanvasTextNotes } from "./useCanvasTextNotes"
import { TopBarSlotProvider } from "@/features/navigation/TopBarSlots"

export interface CanvasEditorProps {
  doc: CanvasDocument
  onChange: (doc: CanvasDocument) => void
  slug?: string
  onImageAdded?: (path: string) => void
  // Path of the `.canvas` file being edited — kept out of the "open canvas" picker so a
  // canvas can't embed itself.
  docPath?: string
  // Nesting depth, set only by canvas windows on themselves. Past a limit a canvas window
  // stops opening its file, so a cycle (A embeds B, B embeds A) can't recurse forever.
  depth?: number
  // Effective zoom of the canvas embedding this one (1 at the top level). The
  // embedded view magnifies with it, so zooming the host scales the nested
  // content along with its frame instead of revealing more or less of it.
  hostScale?: number
}

/** Infinite pan/zoom canvas of ink, frames, attachments and text notes. Camera state is local, every document edit goes through `onChange`. */
export function CanvasEditor({ doc, onChange, slug, onImageAdded, docPath, depth = 0, hostScale = 1 }: CanvasEditorProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  // gates the window/document key listeners below to the frontmost tab
  const active = useTabActive()

  const { scale, offset, settledScale, scaleRef, offsetRef, screenToCanvas, pointAtCenter, isPanning, startPan, movePan, endPan } = useCanvasCamera(containerRef, { viewport: doc.viewport, hostScale, depth })
  const { liveRef, applyChange, commit, snapshot, snapshotOnce, endGesture, undo: undoHistory, redo: redoHistory, canUndo, canRedo } = useCanvasHistory(doc, onChange)
  const { beginDrag, startDrag } = useCanvasDrag({ liveRef, scaleRef, snapshotOnce, applyChange, endGesture })
  const selection = useCanvasSelection({ liveRef, containerRef, scaleRef, offsetRef, commit, beginDrag })
  const { clear: clearSelection, select } = selection
  const insert = useCanvasInsert({ liveRef, commit, pointAtCenter, slug, depth, active, onImageAdded })
  const attachments = useCanvasAttachments({ liveRef, commit, applyChange, slug })
  const captureScreenshot = useCanvasScreenshot({ liveRef, containerRef, screenToCanvas })

  // Selections hold stroke ids, but a restore can still resurrect or drop strokes under the highlight — both directions drop it.
  const undo = useCallback(() => { if (undoHistory()) clearSelection() }, [undoHistory, clearSelection])
  const redo = useCallback(() => { if (redoHistory()) clearSelection() }, [redoHistory, clearSelection])

  const ink = useInkSettings()
  const { tool: inkTool, width: inkWidth, highlighting, color: inkStrokeColor, penColor, clearTool: clearInkTool } = ink
  const rubberBand = useRubberBand(screenToCanvas)
  const { tool, band, cancelBand } = rubberBand
  const { pointerPointRef, autoFocusId, editingTextId, startEditing, editText, focusText, blurText, handleCanvasKeyDown } = useCanvasTextNotes({ liveRef, snapshot, applyChange, endGesture, penColor, pointAtCenter })
  const { spaceDown, pointerIsPen, containerHandlers } = useCanvasPointer({ active, screenToCanvas, pointerPointRef, rubberBand, clearSelection, startPan, movePan, endPan })

  const onBandFinished = useCallback((finished: Band, end: [number, number]) => {
    if (finished.tool === "select") select(finished.points)
    else void captureScreenshot(finished.points[0]!, end)
  }, [select, captureScreenshot])
  const { svgHandlers, livePathRef, isDrawing, isErasing, contextMenu, closeContextMenu } = useInkSurface({
    containerRef, screenToCanvas, scaleRef, liveRef, applyChange, commit, snapshotOnce, endGesture,
    ink, rubberBand, spaceDown, clearSelection, onBandFinished,
  })

  // --- Keyboard: undo/redo, and Escape to cancel a band or drop the stroke selection ---
  useUndoRedoKeys(undo, redo, active)

  useEscapeLayer(() => {
    cancelBand()
    clearSelection()
    clearInkTool()
  }, tool !== "draw" || selection.selectedIds.size > 0 || !!inkTool)

  // A canvas window's body. Passed down rather than imported by the window, so editor and window stay out of an import cycle.
  const renderNestedCanvas = (att: NestedCanvasAttachment, name: string) => (
    <NestedCanvas attachment={att} name={name} slug={slug} depth={depth} hostScale={scale} onChange={(d) => attachments.patchAttachment(att.id, { canvas: d })} />
  )

  const addMenu = (
    <CanvasAddMenu
      slug={slug}
      depth={depth}
      docPath={docPath}
      onAddPage={insert.addPage}
      onAddImage={insert.addImage}
      onAddPdf={insert.addPdf}
      onAddDocument={insert.addDocument}
      onAddArchitecture={insert.addArchitecture}
      onAddNewCanvas={insert.addNewCanvas}
      onAddCanvasFile={insert.addCanvasFile}
      onDocumentRenamed={attachments.followRenamedDocument}
      onDocumentDeleted={attachments.dropDeletedDocument}
    />
  )

  return (
    <div className="flex flex-col h-full">
      <CanvasToolbar
        addMenu={addMenu}
        ink={ink}
        tool={tool}
        onToggleTool={rubberBand.toggleTool}
        isErasing={isErasing}
        selectedCount={selection.selectedIds.size}
        onUndo={undo}
        onRedo={redo}
        canUndo={canUndo}
        canRedo={canRedo}
        zoomPercent={Math.round(scale / hostScale * 100)}
      />

      {/* Canvas area */}
      <div
        ref={containerRef}
        tabIndex={0}
        className={cn("flex-1 min-h-0 overflow-hidden relative", isDrawing || pointerIsPen ? "cursor-none" : isPanning ? "cursor-grabbing" : tool !== "draw" || inkTool ? "cursor-crosshair" : "cursor-grab")}
        style={{ touchAction: "none", WebkitTouchCallout: "none", WebkitUserSelect: "none", userSelect: "none", fontFamily: "var(--font-handwriting), cursive" }}
        onKeyDown={handleCanvasKeyDown}
        {...containerHandlers}
        onContextMenu={(e) => e.preventDefault()}
      >
        <CanvasDotGrid offset={offset} scale={scale} />

        {/* Transformed canvas layer — frames + strokes share this space so strokes
            always render over frame content and both export identically. */}
        <svg className="absolute inset-0 w-full h-full" style={{ touchAction: "none" }} {...svgHandlers}>
          <g transform={`translate(${offset.x},${offset.y}) scale(${scale})`}>
            {doc.frames.map((f) => (
              <CanvasFrameView
                key={f.id}
                frame={f}
                aspect={attachments.frameAspect(f)}
                scale={scale}
                onDragStart={(mode, x, y) => startDrag("frames", f.id, mode, x, y)}
                onRemove={() => attachments.removeFrame(f.id)}
                onAspect={(r) => attachments.setAspect(f.id, r)}
              />
            ))}
            {/* Committed strokes (selected ones are hidden while a drag preview draws them) */}
            <CanvasStrokeLayer strokes={doc.strokes} hidden={selection.preview && selection.selectedIds} />
            {/* In-progress stroke: d is set imperatively by the ink surface */}
            <path ref={livePathRef} fill={inkColor(inkStrokeColor)} className={highlighting ? "ink-highlight" : undefined} />

            {selection.lasso && selection.bounds && (
              <CanvasSelectionOutline
                lasso={selection.lasso}
                bounds={selection.bounds}
                strokes={doc.strokes}
                selectedIds={selection.selectedIds}
                preview={selection.preview}
                scale={scale}
                onDragStart={selection.startDrag}
              />
            )}
            {band?.tool === "select" && <CanvasLassoTrace points={band.points} scale={scale} />}
          </g>
        </svg>

        {selection.bounds && (
          <CanvasSelectionActions bounds={selection.bounds} scale={scale} offset={offset} onCopy={selection.copySelected} onDelete={selection.deleteSelected} />
        )}

        {/* Attachments: live, scrollable, positioned HTML over the SVG, never baked into export.
            Their own scope keeps nested canvases, diagrams and PDFs out of the top bar this canvas uses. */}
        <TopBarSlotProvider>
          {doc.attachments.map((att) => (
            <CanvasAttachmentWindow
              key={att.id}
              attachment={att}
              scale={scale}
              settledScale={settledScale}
              offset={offset}
              pdfAspect={attachments.aspectOf(att.id)}
              full={attachments.fullscreenId === att.id}
              inert={tool === "screenshot"}
              onDragStart={(mode, x, y) => startDrag("attachments", att.id, mode, x, y)}
              onToggleFullscreen={() => attachments.toggleFullscreen(att.id)}
              onRemove={() => attachments.removeAttachment(att.id)}
              onPatch={(patch) => attachments.patchAttachment(att.id, patch)}
              onRetitle={(title) => { void attachments.retitleAttachment(att.id, title) }}
              onPdfAspect={(r) => attachments.setAspect(att.id, r)}
              pdfPen={{ color: inkStrokeColor, width: inkWidth, highlight: highlighting }}
              renderNestedCanvas={renderNestedCanvas}
            />
          ))}
        </TopBarSlotProvider>

        {/* Screenshot selection belongs above live attachments so it remains visible while dragging over a PDF. */}
        {band?.tool === "screenshot" && <CanvasScreenshotBox from={band.points[0]!} to={band.points[band.points.length - 1]!} scale={scale} offset={offset} />}

        {/* Text notes render their Markdown until clicked, then expose the source textarea.
            They remain rasterized as source text in canvas exports. */}
        {doc.texts.map((t) => (
          <CanvasTextNote
            key={t.id}
            note={t}
            scale={scale}
            offset={offset}
            editing={t.id === editingTextId}
            autoFocus={t.id === autoFocusId}
            onStartEditing={() => startEditing(t.id)}
            onEdit={(value) => editText(t.id, value)}
            onFocusNote={() => focusText(t.id)}
            onBlurNote={() => blurText(t.id)}
            onDragStart={(mode, x, y) => startDrag("texts", t.id, mode, x, y)}
          />
        ))}

        {contextMenu && (
          <CanvasContextMenu at={contextMenu} canPaste={!!slug} onClose={closeContextMenu} onPasteImage={(x, y) => { void insert.pasteImageAtPoint(x, y) }} />
        )}
      </div>
    </div>
  )
}

const MAX_NEST_DEPTH = 2 // canvas windows stop opening files here, so A→B→A can't recurse

// Stable fallback for a nested canvas saved without contents, a fresh object per
// render would reset the nested editor's identity on every parent re-render.
const EMPTY_NESTED = emptyCanvasDoc()

/**
 * Body of a canvas window on a canvas at `depth`: inline contents edit through `onChange`, a canvas file opens its own collaboration.
 * Past MAX_NEST_DEPTH a window stops opening its file, so a cycle (A embeds B, B embeds A) can't recurse forever.
 */
function NestedCanvas({ attachment, name, slug, depth, hostScale, onChange }: {
  attachment: NestedCanvasAttachment
  name: string
  slug?: string
  depth: number
  hostScale: number
  onChange: (doc: CanvasDocument) => void
}) {
  if (!attachment.path) {
    return <CanvasEditor doc={attachment.canvas ?? EMPTY_NESTED} onChange={onChange} slug={slug} depth={depth + 1} hostScale={hostScale} />
  }
  if (depth >= MAX_NEST_DEPTH) {
    return (
      <div className="flex h-full items-center justify-center px-3 text-center text-[10px] text-ink-faint">
        {name} — open it directly to edit
      </div>
    )
  }
  return <NestedCanvasFile path={attachment.path} slug={slug} depth={depth + 1} hostScale={hostScale} />
}

// A file-backed window is another view onto the same canvas collaboration owner.
function NestedCanvasFile({ path, slug, depth, hostScale }: {
  path: string
  slug?: string
  depth: number
  hostScale: number
}) {
  const { document, replace, status, retry } = useCollaborativeCanvas(path)

  if (!document) {
    return status === "error"
      ? <LoadFailure message="Failed to load this canvas." onRetry={retry} size="fill" />
      : <LoadingPlaceholder size="fill" />
  }
  return <CanvasEditor doc={document} onChange={replace} slug={slug} depth={depth} hostScale={hostScale} docPath={path} />
}
