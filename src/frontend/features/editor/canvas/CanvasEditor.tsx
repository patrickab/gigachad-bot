"use client"

import { useCallback, useRef, useState } from "react"
import { ImageIcon } from "lucide-react"
import { A4_ASPECT, FALLBACK_ASPECT, TEXT_DEFAULT_HEIGHT, TEXT_DEFAULT_WIDTH, canvasEntityId, emptyCanvasDoc, omitKey, type CanvasAttachment, type CanvasDocument, type CanvasFrame } from "@/lib/domain/canvasDocument"
import { imageFrameEmbeds } from "@/lib/domain/canvasRender"
import { ERASER_RADIUS_PX, appendPointerSamples, finishStroke, inkColor, newStroke, strokesAfterErase } from "@/features/editor/canvas/ink/canvasInk"
import type { Poly } from "@/features/editor/canvas/selection/canvasSelection"
import { inkOutlinePath, penPressure, renderPageToPng, type CanvasEmbedRect } from "@/lib/drawing"
import { renameDocument } from "@/lib/backend/api"
import { useCollaborativeCanvas } from "@/features/editor/canvas/useCollaborativeCanvas"
import { useCanvasCamera } from "@/features/editor/canvas/camera/useCanvasCamera"
import { useCanvasDrag } from "@/features/editor/canvas/selection/useCanvasDrag"
import { useCanvasHistory, useUndoRedoKeys } from "@/features/editor/canvas/useCanvasHistory"
import { useCanvasInsert } from "@/features/editor/canvas/attachments/useCanvasInsert"
import { useTabActive } from "@/contexts/TabContext"
import { useCanvasSelection } from "@/features/editor/canvas/selection/useCanvasSelection"
import { useEventListener } from "@/hooks/useEventListener"
import { LoadFailure, LoadingPlaceholder } from "@/components/ui/DocumentLoadState"
import { cn, isEditableTarget } from "@/lib/utils"
import { CanvasAddMenu } from "@/features/editor/canvas/attachments/CanvasAddMenu"
import { CanvasAttachmentWindow } from "@/features/editor/canvas/attachments/CanvasAttachmentWindow"
import { CanvasDotGrid } from "@/features/editor/canvas/camera/CanvasDotGrid"
import { CanvasFrameView } from "@/features/editor/canvas/attachments/CanvasFrameView"
import { CanvasLassoTrace, CanvasScreenshotBox, CanvasSelectionActions, CanvasSelectionOutline } from "@/features/editor/canvas/selection/CanvasSelectionOverlay"
import { CanvasStrokeLayer } from "@/features/editor/canvas/ink/CanvasStrokeLayer"
import { CanvasTextNote } from "./CanvasTextNote"
import { CanvasToolbar, useInkSettings, type CanvasTool, type CanvasToolbarSlots } from "./CanvasToolbar"

const MAX_NEST_DEPTH = 2 // canvas windows stop opening files here, so A→B→A can't recurse
const PEN_HOLD_MS = 400 // press-and-hold the pen button to activate selection mode
const PEN_DOUBLE_CLICK_MS = 350 // double-click the pen button to activate screenshot mode
const TEXT_DEFAULT_SIZE = 28 // canvas units — font size for new text notes
const STRAIGHTEN_HOLD_MS = 500 // hold the pen still mid-stroke to snap it into a straight line
const STRAIGHTEN_MOVE_TOLERANCE = 5 // px of client movement that resets the "holding still" timer
const LONG_PRESS_MS = 500 // touch/mouse press-and-hold opens the context menu
const LONG_PRESS_TOLERANCE = 5 // px of client movement that cancels a long press
const SCREENSHOT_MIN = 4 // canvas units — smaller drags are accidents

// Stable fallback for a nested canvas saved without contents — a fresh object per
// render would reset the nested editor's identity on every parent re-render.
const EMPTY_NESTED = emptyCanvasDoc()

interface CanvasEditorProps {
  doc: CanvasDocument
  onChange: (doc: CanvasDocument) => void
  slug?: string
  onImageAdded?: (path: string) => void
  // When given, the toolbar renders into these external elements (the app header pills
  // in canvas mode) instead of above the drawing surface.
  toolbarSlots?: CanvasToolbarSlots | null
  // Path of the `.canvas` file being edited — kept out of the "open canvas" picker so a
  // canvas can't embed itself.
  docPath?: string
  // Nesting depth, set only by canvas windows on themselves. Past MAX_NEST_DEPTH a
  // canvas window stops opening its file, so a cycle (A embeds B, B embeds A) can't
  // recurse forever.
  depth?: number
  // Effective zoom of the canvas embedding this one (1 at the top level). The
  // embedded view magnifies with it, so zooming the host scales the nested
  // content along with its frame instead of revealing more or less of it.
  hostScale?: number
}

/** A rubber-band drag in canvas units: the lasso traces every point, the screenshot rectangle uses its first and last. */
type Band = { tool: Exclude<CanvasTool, "draw">; points: Poly }

/** Infinite pan/zoom canvas of ink, frames, attachments and text notes. Camera state is local, every document edit goes through `onChange`. */
export function CanvasEditor({ doc, onChange, slug, onImageAdded, toolbarSlots, docPath, depth = 0, hostScale = 1 }: CanvasEditorProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  // gates the window/document key listeners below to the frontmost tab
  const active = useTabActive()

  const { scale, offset, settledScale, scaleRef, offsetRef, screenToCanvas, pointAtCenter, isPanning, startPan, movePan, endPan } = useCanvasCamera(containerRef, { viewport: doc.viewport, hostScale, depth })
  const { liveRef, applyChange, commit, snapshot, snapshotOnce, endGesture, undo: undoHistory, redo: redoHistory, canUndo, canRedo } = useCanvasHistory(doc, onChange)
  const selection = useCanvasSelection({ liveRef, containerRef, scaleRef, offsetRef, commit })
  const { clear: clearSelection } = selection
  const startDrag = useCanvasDrag({ liveRef, scaleRef, snapshotOnce, applyChange, endGesture, moveSelectionDrag: selection.moveDrag, endSelectionDrag: selection.endDrag })
  const insert = useCanvasInsert({ liveRef, commit, pointAtCenter, slug, depth, active, onImageAdded })

  // Selections hold stroke ids, but a restore can still resurrect or drop strokes under the highlight — both directions drop it.
  const undo = useCallback(() => { if (undoHistory()) clearSelection() }, [undoHistory, clearSelection])
  const redo = useCallback(() => { if (redoHistory()) clearSelection() }, [redoHistory, clearSelection])

  const [tool, setTool] = useState<CanvasTool>("draw")
  const ink = useInkSettings()
  const { tool: inkTool, width: inkWidth, highlighting, color: inkStrokeColor, penColor, clearTool: clearInkTool } = ink

  // Hide the cursor for pen input without updating state on every move.
  const [pointerIsPen, setPointerIsPen] = useState(false)
  const lastPointerTypeRef = useRef<string>("mouse")

  const [isDrawing, setIsDrawing] = useState(false)
  // The in-progress stroke lives in a ref and renders imperatively via livePathRef,
  // so a pointermove never triggers a React re-render of the whole editor.
  const currentPointsRef = useRef<number[][]>([])
  const livePathRef = useRef<SVGPathElement>(null)
  // Pointer capture can still deliver a late event from another active device. Only
  // the pointer that began a stroke may extend or finish it.
  const drawingPointerId = useRef<number | null>(null)
  const [isErasing, setIsErasing] = useState(false)

  const redrawLive = useCallback(() => {
    const el = livePathRef.current
    if (!el) return
    const pts = currentPointsRef.current
    el.setAttribute("d", pts.length >= 2 ? inkOutlinePath(pts, inkWidth, highlighting) : "")
  }, [inkWidth, highlighting])

  // --- Rubber band: a lasso (select) or a rectangle (screenshot) dragged on the canvas.
  // The ref is the synchronous truth for the pointer handlers; the state drives rendering. ---
  const [band, setBand] = useState<Band | null>(null)
  const bandRef = useRef<Band | null>(null)
  const writeBand = useCallback((next: Band | null) => {
    bandRef.current = next
    setBand(next)
  }, [])
  const startBand = useCallback((bandTool: Band["tool"], point: [number, number]) => writeBand({ tool: bandTool, points: [point] }), [writeBand])
  const extendBand = useCallback((point: [number, number]) => {
    const b = bandRef.current
    if (b) writeBand({ tool: b.tool, points: b.tool === "select" ? [...b.points, point] : [b.points[0]!, point] })
  }, [writeBand])

  const toggleTool = useCallback((next: Band["tool"]) => setTool((current) => current === next ? "draw" : next), [])

  // Pen side-button gesture: hold => activate selection mode, double-click => activate screenshot mode
  const penHoldTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const penHoldFired = useRef(false)
  const penClickPending = useRef(false)
  const penClickTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const screenshotSequence = useRef(0)

  // --- Text: notes render as Markdown when unfocused. Enter switches from source
  // to formatted content, while clicking the rendered note reopens its textarea. ---
  const lastCanvasPoint = useRef<[number, number] | null>(null)
  const newTextIds = useRef(new Set<string>())
  const [autoFocusId, setAutoFocusId] = useState<string | null>(null)
  const [editingTextId, setEditingTextId] = useState<string | null>(null)
  // One history gesture per focus session of a text note, closed on blur, so typing a
  // sentence is one undo step, not one per keystroke. A freshly created box opens no new
  // one — its creation gesture (from addText) stays open and covers the typing.
  const textEdited = useRef(false)

  const addText = useCallback((x: number, y: number, width = TEXT_DEFAULT_WIDTH, height = TEXT_DEFAULT_HEIGHT, text = "") => {
    const id = canvasEntityId()
    newTextIds.current.add(id)
    const cur = liveRef.current.doc
    // The gesture stays open until the note blurs, so what is typed into it undoes together with its creation.
    snapshot()
    applyChange({ ...cur, texts: [...cur.texts, { id, x, y, width, height, text, color: penColor, size: TEXT_DEFAULT_SIZE }] })
    setAutoFocusId(id)
    setEditingTextId(id)
  }, [liveRef, penColor, snapshot, applyChange])

  const editText = useCallback((id: string, text: string) => {
    const cur = liveRef.current.doc
    if (!textEdited.current) {
      textEdited.current = true
      if (cur.texts.find((t) => t.id === id)?.text !== "") snapshot()
    }
    applyChange({ ...cur, texts: cur.texts.map((t) => t.id === id ? { ...t, text } : t) })
  }, [liveRef, snapshot, applyChange])

  const focusText = useCallback((id: string) => {
    textEdited.current = newTextIds.current.delete(id)
    setAutoFocusId((cur) => cur === id ? null : cur)
  }, [])

  // Removes a text note if its content is blank — called on blur so an
  // untouched or fully-cleared box doesn't linger as an empty artifact.
  const blurText = useCallback((id: string) => {
    setEditingTextId(null)
    const cur = liveRef.current.doc
    const item = cur.texts.find((t) => t.id === id)
    if (item && item.text.trim() === "") {
      applyChange({ ...cur, texts: cur.texts.filter((t) => t.id !== id) })
    }
    endGesture()
  }, [liveRef, applyChange, endGesture])

  const handleCanvasKeyDown = useCallback((e: React.KeyboardEvent<HTMLDivElement>) => {
    // Only a single printable character starts a note (named keys like "Enter" are longer).
    if (e.ctrlKey || e.metaKey || e.altKey || e.nativeEvent.isComposing || Array.from(e.key).length !== 1) return
    // Only the bare canvas surface should turn typing into a new text note.
    // Any nested editable (architecture diagram fields, modals, contenteditable) keeps its keystrokes.
    if (isEditableTarget(e.target, { includeSelect: true })) return

    e.preventDefault()
    const centered = pointAtCenter(TEXT_DEFAULT_WIDTH, TEXT_DEFAULT_HEIGHT / TEXT_DEFAULT_WIDTH)
    const [x, y] = lastCanvasPoint.current ?? [centered.cx, centered.cy]
    addText(x, y, TEXT_DEFAULT_WIDTH, TEXT_DEFAULT_HEIGHT, e.key)
  }, [addText, pointAtCenter])

  // --- Aspect cache (image frames + pdf attachments), keyed by element id ---
  const [aspects, setAspects] = useState<Record<string, number>>({})
  const setAspect = useCallback((id: string, r: number) => {
    setAspects((prev) => (Math.abs((prev[id] ?? 0) - r) < 0.001 ? prev : { ...prev, [id]: r }))
  }, [])
  const aspectFor = useCallback((f: CanvasFrame) => f.kind === "page" ? A4_ASPECT : (aspects[f.id] ?? FALLBACK_ASPECT), [aspects])

  /** Commits the document without entity `id` of `collection`, dropping its measured aspect. */
  const removeEntity = useCallback((collection: "frames" | "attachments", id: string) => {
    const cur = liveRef.current.doc
    commit({ ...cur, [collection]: cur[collection].filter((entity) => entity.id !== id) })
    setAspects((prev) => omitKey(prev, id))
  }, [liveRef, commit])
  const removeFrame = useCallback((id: string) => removeEntity("frames", id), [removeEntity])

  // --- Attachment windows ---
  // Which attachment window, if any, is blown up over the whole surface. Exiting drops
  // straight back onto the canvas that holds it.
  const [fullscreenId, setFullscreenId] = useState<string | null>(null)

  const removeAttachment = useCallback((id: string) => {
    removeEntity("attachments", id)
    setFullscreenId((fullscreen) => (fullscreen === id ? null : fullscreen))
  }, [removeEntity])

  /** Applies `edit` to the live attachments; nothing is written when it hands them back unchanged. */
  const editAttachments = useCallback((edit: (all: CanvasAttachment[]) => CanvasAttachment[]) => {
    const cur = liveRef.current.doc
    const attachments = edit(cur.attachments)
    if (attachments !== cur.attachments) applyChange({ ...cur, attachments })
  }, [liveRef, applyChange])

  // Navigation, window state and titles (the tree's file, the PDF page, the connection style) persist without an undo step.
  // ponytail: nested canvas edits bypass the parent's history too — the nested editor has its own
  // undo/redo, and snapshotting the whole parent per nested stroke would be absurd.
  const patchAttachment = useCallback((id: string, patch: Partial<CanvasAttachment>) => editAttachments((all) => {
    const attachment = all.find((a) => a.id === id)
    if (!attachment || Object.entries(patch).every(([key, value]) => attachment[key as keyof CanvasAttachment] === value)) return all
    return all.map((a) => a.id === id ? { ...a, ...patch } : a)
  }), [editAttachments])

  const retitleAttachment = useCallback(async (id: string, nextTitle: string) => {
    const attachment = liveRef.current.doc.attachments.find((a) => a.id === id)
    const title = nextTitle.trim()
    if (!attachment || !title || attachment.title === title) return
    // Architecture sources have no rename, so renaming an architecture window only retitles the window.
    if (!attachment.path || attachment.kind === "architecture-graph") { patchAttachment(id, { title }); return }
    try {
      const renamed = await renameDocument(slug ?? "", attachment.path, title)
      patchAttachment(id, { path: renamed.path, title })
    } catch { /* the stored name stays authoritative when rename fails */ }
  }, [liveRef, patchAttachment, slug])

  // Attachments on this canvas follow a file the add menu renames; they go once its delete commits, since the file is then gone for good.
  const followRenamedDocument = useCallback((oldPath: string, newPath: string) => {
    editAttachments((all) => all.some((a) => a.path === oldPath) ? all.map((a) => a.path === oldPath ? { ...a, path: newPath } : a) : all)
  }, [editAttachments])

  const dropDeletedDocument = useCallback((path: string) => {
    const gone = liveRef.current.doc.attachments.filter((a) => a.path === path).map((a) => a.id)
    if (gone.length === 0) return
    editAttachments((all) => all.filter((a) => a.path !== path))
    setFullscreenId((id) => (id && gone.includes(id) ? null : id))
  }, [liveRef, editAttachments])

  // A canvas window's body. Passed down rather than imported by the window, so editor and window stay out of an import cycle.
  const renderNestedCanvas = (att: CanvasAttachment, name: string) => {
    if (!att.path) {
      return (
        <CanvasEditor
          doc={att.canvas ?? EMPTY_NESTED}
          onChange={(d) => patchAttachment(att.id, { canvas: d })}
          slug={slug}
          depth={depth + 1}
          hostScale={scale}
        />
      )
    }
    if (depth >= MAX_NEST_DEPTH) {
      return (
        <div className="flex h-full items-center justify-center px-3 text-center text-[10px] text-ink-faint">
          {name} — open it directly to edit
        </div>
      )
    }
    return <NestedCanvasFile path={att.path} slug={slug} depth={depth + 1} hostScale={scale} />
  }

  // --- Long-press context menu ---
  const [contextMenu, setContextMenu] = useState<{ left: number; top: number; canvasX: number; canvasY: number } | null>(null)
  const longPressTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const longPressPos = useRef<{ clientX: number; clientY: number } | null>(null)

  // --- Straighten-on-hold: pausing mid-stroke for STRAIGHTEN_HOLD_MS snaps the
  // in-progress freehand stroke to a straight line from its start to the hold point. ---
  const straightenTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const straightenAnchor = useRef<{ clientX: number; clientY: number } | null>(null)
  const straightened = useRef(false)

  const armStraighten = useCallback((clientX: number, clientY: number) => {
    straightenAnchor.current = { clientX, clientY }
    clearTimeout(straightenTimer.current)
    straightenTimer.current = setTimeout(() => {
      const pts = currentPointsRef.current
      if (pts.length < 2) return
      straightened.current = true
      currentPointsRef.current = [pts[0]!, pts[pts.length - 1]!]
      redrawLive()
    }, STRAIGHTEN_HOLD_MS)
  }, [redrawLive])

  const cancelStraighten = useCallback(() => {
    clearTimeout(straightenTimer.current)
    straightenAnchor.current = null
    straightened.current = false
  }, [])

  // Drop the in-progress stroke, e.g. the tiny hold-dot a long press leaves behind.
  const abortStroke = useCallback(() => {
    setIsDrawing(false)
    currentPointsRef.current = []
    drawingPointerId.current = null
    redrawLive()
    cancelStraighten()
  }, [redrawLive, cancelStraighten])

  // Fires the context menu after LONG_PRESS_MS without a cancelling move or release.
  const startLongPress = useCallback((clientX: number, clientY: number) => {
    longPressPos.current = { clientX, clientY }
    longPressTimer.current = setTimeout(() => {
      const [cx, cy] = screenToCanvas(clientX, clientY)
      abortStroke()
      const containerRect = containerRef.current?.getBoundingClientRect()
      setContextMenu({ left: clientX - (containerRect?.left ?? 0), top: clientY - (containerRect?.top ?? 0), canvasX: cx, canvasY: cy })
    }, LONG_PRESS_MS)
  }, [screenToCanvas, abortStroke])

  const cancelLongPress = useCallback(() => {
    clearTimeout(longPressTimer.current)
    longPressPos.current = null
  }, [])

  useEventListener("document", "pointerdown", () => setContextMenu(null), { active: !!contextMenu })
  useEventListener("document", "keydown", (e) => { if (e.key === "Escape") setContextMenu(null) }, { active: !!contextMenu })

  // --- Screenshot capture: rasterize a canvas-space rect and copy it to the clipboard ---
  const captureScreenshot = useCallback(async ([x0, y0]: [number, number], [x1, y1]: [number, number]) => {
    const x = Math.min(x0, x1)
    const y = Math.min(y0, y1)
    const w = Math.abs(x1 - x0)
    const h = Math.abs(y1 - y0)
    if (w < SCREENSHOT_MIN || h < SCREENSHOT_MIN) return
    const sequence = ++screenshotSequence.current
    const { frames, strokes, texts } = liveRef.current.doc
    const images = imageFrameEmbeds(frames, aspectFor)
    const pdfPages: CanvasEmbedRect[] = Array.from(
      containerRef.current?.querySelectorAll<HTMLCanvasElement>("[data-canvas-attachment] .react-pdf__Page__canvas") ?? [],
    ).flatMap((image) => {
      const rect = image.getBoundingClientRect()
      if (rect.width === 0 || rect.height === 0) return []
      const [left, top] = screenToCanvas(rect.left, rect.top)
      const [right, bottom] = screenToCanvas(rect.right, rect.bottom)
      return [{ image, x: left, y: top, width: right - left, aspect: (bottom - top) / (right - left) }]
    })
    try {
      const pngBytes = await renderPageToPng(strokes, x, y, w, h, images, texts, pdfPages)
      if (sequence !== screenshotSequence.current) return
      const blob = new Blob([pngBytes.buffer as ArrayBuffer], { type: "image/png" })
      await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })])
    } catch { /* clipboard write can fail without permission — silently drop */ }
  }, [liveRef, aspectFor, screenToCanvas])

  // --- Container pointer handling: pan, pen side-button and typing position ---
  const spaceDown = useRef(false)
  useEventListener("window", "keydown", (e) => { if (e.code === "Space" && !e.repeat) spaceDown.current = true }, { active })
  useEventListener("window", "keyup", (e) => { if (e.code === "Space") spaceDown.current = false }, { active })

  const handleContainerPointerDown = useCallback((e: React.PointerEvent) => {
    if (e.pointerType === "touch") return
    ;(e.currentTarget as HTMLElement).focus({ preventScroll: true })
    lastCanvasPoint.current = screenToCanvas(e.clientX, e.clientY)
    if (tool !== "draw") return
    // any pointerdown that bubbles this far didn't hit the selection lasso or its
    // handles (they stopPropagation) — so it's "outside" and drops the selection
    clearSelection()
    if (e.button === 1 || (e.pointerType === "pen" && e.button === 2)) {
      // pen side button: hold = activate selection mode & start the lasso right on this
      // same press (no lift-and-retouch needed), double-click = activate screenshot mode
      e.preventDefault()
      ;(e.target as Element).setPointerCapture(e.pointerId)
      penHoldFired.current = false
      clearTimeout(penHoldTimer.current)
      const clientX = e.clientX, clientY = e.clientY
      penHoldTimer.current = setTimeout(() => {
        penHoldFired.current = true
        setTool("select")
        startBand("select", screenToCanvas(clientX, clientY))
      }, PEN_HOLD_MS)
      return
    }
    if ((e.button === 0 && (spaceDown.current || e.pointerType === "mouse")) || (e.button === 2 && e.pointerType === "mouse")) startPan(e)
  }, [tool, clearSelection, screenToCanvas, startBand, startPan])

  const handleContainerPointerMove = useCallback((e: React.PointerEvent) => {
    if (e.pointerType === "touch") return
    lastCanvasPoint.current = screenToCanvas(e.clientX, e.clientY)
    if (e.pointerType !== lastPointerTypeRef.current) {
      lastPointerTypeRef.current = e.pointerType
      setPointerIsPen(e.pointerType === "pen")
    }
    movePan(e)
  }, [screenToCanvas, movePan])

  const handleContainerPointerUp = useCallback((e: React.PointerEvent) => {
    if (e.pointerType === "touch") return
    if (e.button === 1 || (e.pointerType === "pen" && e.button === 2)) {
      clearTimeout(penHoldTimer.current)
      if (penHoldFired.current) { penHoldFired.current = false; return }
      if (penClickPending.current) {
        penClickPending.current = false
        clearTimeout(penClickTimer.current)
        setTool("screenshot")
      } else {
        penClickPending.current = true
        penClickTimer.current = setTimeout(() => { penClickPending.current = false }, PEN_DOUBLE_CLICK_MS)
      }
      return
    }
    endPan()
  }, [endPan])

  // --- Drawing (on SVG) ---
  const handleSvgPointerDown = useCallback((e: React.PointerEvent<SVGSVGElement>) => {
    if (e.pointerType === "touch") return
    if (spaceDown.current) return
    if (tool !== "draw") {
      if (e.button !== 0) return
      e.stopPropagation()
      e.currentTarget.setPointerCapture(e.pointerId)
      startBand(tool, screenToCanvas(e.clientX, e.clientY))
      return
    }
    // drawing/erasing directly on the canvas counts as "outside" the selection
    clearSelection()
    // The mouse draws only with a selected ink tool, otherwise its drag falls through to the container and pans.
    if (e.pointerType === "mouse" && !inkTool) return
    // button 5 is the pen's eraser end
    if (e.button === 5) { setIsErasing(true); return }
    if (e.button !== 0) return
    e.stopPropagation()
    if (inkTool === "eraser") { setIsErasing(true); return }
    e.currentTarget.setPointerCapture(e.pointerId)
    setIsDrawing(true)
    const [cx, cy] = screenToCanvas(e.clientX, e.clientY)
    currentPointsRef.current = [[cx, cy, penPressure(e)]]
    drawingPointerId.current = e.pointerId
    redrawLive()
    // the long-press context menu (incl. "Paste image") is a touch affordance — a pen or
    // mouse stroke that pauses mid-draw must never be mistaken for a long-press
    if (e.pointerType !== "pen" && e.pointerType !== "mouse") startLongPress(e.clientX, e.clientY)
    armStraighten(e.clientX, e.clientY)
  }, [tool, inkTool, screenToCanvas, startBand, startLongPress, armStraighten, clearSelection, redrawLive])

  const handleSvgPointerMove = useCallback((e: React.PointerEvent<SVGSVGElement>) => {
    if (e.pointerType === "touch") return
    if (bandRef.current) {
      extendBand(screenToCanvas(e.clientX, e.clientY))
      return
    }
    // cancel long-press once the pointer moves more than LONG_PRESS_TOLERANCE px
    if (longPressPos.current) {
      const dx = e.clientX - longPressPos.current.clientX
      const dy = e.clientY - longPressPos.current.clientY
      if (dx * dx + dy * dy > LONG_PRESS_TOLERANCE * LONG_PRESS_TOLERANCE) cancelLongPress()
    }
    if (isErasing) {
      const [ex, ey] = screenToCanvas(e.clientX, e.clientY)
      const cur = liveRef.current.doc
      const updated = strokesAfterErase(cur.strokes, ex, ey, ERASER_RADIUS_PX / scaleRef.current)
      if (updated.length !== cur.strokes.length) {
        snapshotOnce()
        applyChange({ ...cur, strokes: updated })
      }
      return
    }
    if (!isDrawing || e.pointerId !== drawingPointerId.current) return
    if (straightened.current) {
      const [cx, cy] = screenToCanvas(e.clientX, e.clientY)
      currentPointsRef.current = [currentPointsRef.current[0]!, [cx, cy, penPressure(e)]]
      redrawLive()
      return
    }
    if (straightenAnchor.current) {
      const dx = e.clientX - straightenAnchor.current.clientX
      const dy = e.clientY - straightenAnchor.current.clientY
      if (dx * dx + dy * dy > STRAIGHTEN_MOVE_TOLERANCE * STRAIGHTEN_MOVE_TOLERANCE) armStraighten(e.clientX, e.clientY)
    }
    appendPointerSamples(currentPointsRef.current, e.nativeEvent, screenToCanvas)
    redrawLive()
  }, [isDrawing, isErasing, liveRef, scaleRef, screenToCanvas, extendBand, cancelLongPress, armStraighten, redrawLive, snapshotOnce, applyChange])

  const handleSvgPointerUp = useCallback((e: React.PointerEvent<SVGSVGElement>) => {
    cancelLongPress()
    cancelStraighten()
    if (e.pointerType === "touch") return
    const finished = bandRef.current
    if (finished) {
      writeBand(null)
      setTool("draw")
      if (finished.tool === "select") selection.select(finished.points)
      else void captureScreenshot(finished.points[0]!, screenToCanvas(e.clientX, e.clientY))
      return
    }
    if (isErasing) {
      setIsErasing(false)
      endGesture()
      return
    }
    if (!isDrawing || e.pointerId !== drawingPointerId.current) return
    setIsDrawing(false)
    const pts = finishStroke(currentPointsRef.current, e, screenToCanvas)
    currentPointsRef.current = []
    drawingPointerId.current = null
    redrawLive()
    if (!pts) return
    const cur = liveRef.current.doc
    commit({ ...cur, strokes: [...cur.strokes, newStroke(pts, { color: inkStrokeColor, width: inkWidth, highlight: highlighting })] })
  }, [isDrawing, isErasing, inkStrokeColor, inkWidth, highlighting, liveRef, selection, cancelLongPress, cancelStraighten, captureScreenshot, writeBand, screenToCanvas, redrawLive, commit, endGesture])

  // --- Keyboard: undo/redo, and Escape to cancel a band or drop the stroke selection ---
  useUndoRedoKeys(undo, redo, active)

  useEventListener("document", "keydown", (e) => {
    if (e.key !== "Escape") return
    writeBand(null)
    setTool("draw")
    clearSelection()
    clearInkTool()
  }, { active: tool !== "draw" || selection.selectedIds.size > 0 || !!inkTool })

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
      onDocumentRenamed={followRenamedDocument}
      onDocumentDeleted={dropDeletedDocument}
    />
  )

  return (
    <div className="flex flex-col h-full">
      <CanvasToolbar
        slots={toolbarSlots}
        addMenu={addMenu}
        ink={ink}
        tool={tool}
        onToggleTool={toggleTool}
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
        onPointerDown={handleContainerPointerDown}
        onPointerMove={handleContainerPointerMove}
        onPointerUp={handleContainerPointerUp}
        onContextMenu={(e) => e.preventDefault()}
      >
        <CanvasDotGrid offset={offset} scale={scale} />

        {/* Transformed canvas layer — frames + strokes share this space so strokes
            always render over frame content and both export identically. */}
        <svg
          className="absolute inset-0 w-full h-full"
          style={{ touchAction: "none" }}
          onPointerDown={handleSvgPointerDown}
          onPointerMove={handleSvgPointerMove}
          onPointerUp={handleSvgPointerUp}
          onPointerCancel={handleSvgPointerUp}
          onLostPointerCapture={handleSvgPointerUp}
        >
          <g transform={`translate(${offset.x},${offset.y}) scale(${scale})`}>
            {doc.frames.map((f) => (
              <CanvasFrameView
                key={f.id}
                frame={f}
                aspect={aspectFor(f)}
                scale={scale}
                onDragStart={(mode, x, y) => startDrag("frames", f.id, mode, x, y)}
                onRemove={() => removeFrame(f.id)}
                onAspect={(r) => setAspect(f.id, r)}
              />
            ))}
            {/* Committed strokes (selected ones are hidden while a drag preview draws them) */}
            <CanvasStrokeLayer strokes={doc.strokes} hidden={selection.preview && selection.selectedIds} />
            {/* In-progress stroke — d is set imperatively by redrawLive */}
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

        {/* Attachments — live, scrollable, positioned HTML over the SVG, never baked into export. */}
        {doc.attachments.map((att) => (
          <CanvasAttachmentWindow
            key={att.id}
            attachment={att}
            scale={scale}
            settledScale={settledScale}
            offset={offset}
            pdfAspect={aspects[att.id] ?? FALLBACK_ASPECT}
            full={fullscreenId === att.id}
            inert={tool === "screenshot"}
            onDragStart={(mode, x, y) => startDrag("attachments", att.id, mode, x, y)}
            onToggleFullscreen={() => setFullscreenId(fullscreenId === att.id ? null : att.id)}
            onRemove={() => removeAttachment(att.id)}
            onPatch={(patch) => patchAttachment(att.id, patch)}
            onRetitle={(title) => { void retitleAttachment(att.id, title) }}
            onPdfAspect={(r) => setAspect(att.id, r)}
            pdfPen={{ color: inkStrokeColor, width: inkWidth, highlight: highlighting }}
            renderNestedCanvas={renderNestedCanvas}
          />
        ))}

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
            onStartEditing={() => { setEditingTextId(t.id); setAutoFocusId(t.id) }}
            onEdit={(text) => editText(t.id, text)}
            onFocusNote={() => focusText(t.id)}
            onBlurNote={() => blurText(t.id)}
            onDragStart={(mode, x, y) => startDrag("texts", t.id, mode, x, y)}
          />
        ))}

        {/* Long-press context menu */}
        {contextMenu && (
          <div
            className="absolute z-20 rounded-lg border border-divider bg-paper shadow-[var(--shadow-lg)] py-1 min-w-[100px]"
            style={{ left: contextMenu.left, top: contextMenu.top }}
            onPointerDown={(e) => e.stopPropagation()}
          >
            <button
              onClick={() => {
                const { canvasX, canvasY } = contextMenu
                setContextMenu(null)
                void insert.pasteImageAtPoint(canvasX, canvasY)
              }}
              disabled={!slug}
              className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-[11px] text-ink-muted hover:text-ink hover:bg-hover transition-colors disabled:opacity-30"
            >
              <ImageIcon className="h-3 w-3 shrink-0 text-ink-faint" />
              Paste image
            </button>
          </div>
        )}
      </div>
    </div>
  )
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
  return (
    <CanvasEditor
      doc={document}
      onChange={replace}
      slug={slug}
      depth={depth}
      hostScale={hostScale}
      docPath={path}
    />
  )
}
