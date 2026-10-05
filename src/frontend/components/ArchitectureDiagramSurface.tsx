"use client"

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type CSSProperties, type Dispatch, type FocusEvent, type KeyboardEvent, type PointerEvent as ReactPointerEvent, type ReactNode, type RefObject, type SetStateAction, type WheelEvent as ReactWheelEvent } from "react"
import {
  Background, BaseEdge, ConnectionMode, Handle, NodeResizer, Position, ReactFlow, useEdgesState, useInternalNode, useNodesState,
  type Connection, type Edge, type EdgeProps, type InternalNode, type Node, type NodeChange, type NodeProps, type OnBeforeDelete, type OnConnect, type OnConnectEnd, type ReactFlowInstance,
} from "@xyflow/react"
import "@xyflow/react/dist/style.css"
import rough from "roughjs"
import { CircleDashed, Maximize, MoreHorizontal, Plus, RotateCcw, Trash2, X } from "lucide-react"
import { useClickOutside } from "@/hooks/useClickOutside"
import { useTwoFingerGesture } from "@/hooks/useTwoFingerGesture"
import { pointInPolygon } from "@/lib/drawing"
import { cn } from "@/lib/utils"
import { useTabActive } from "./TabManager"
import { reframeCamera } from "./InfiniteViewport"
import { FALLBACK_SIZE, isDraft, nextDraftId, nodeMinWidth, NODE_MIN_WIDTH, type DiagramEdge, type DiagramGraph, type DiagramNode, type EdgeStyle } from "@/lib/architecture"
import { StyledSelect } from "./StyledSelect"

interface EdgeRuntime {
  onPathChange?: (id: string, path?: DiagramEdge["path"]) => void
  onLabelChange?: (id: string, label: string) => void
  toFlowPoint?: (x: number, y: number) => { x: number, y: number } | undefined
}
interface GraphFlowEdgeData extends DiagramEdge, EdgeRuntime, Record<string, unknown> {
  attachment?: EdgeAttachment
  edgeStyle: EdgeStyle
}
type GraphFlowNode = Node<ArchitectureNodeData, "architecture-node">
type GraphFlowEdge = Edge<GraphFlowEdgeData, "architecture-edge">
type GraphFlowInstance = ReactFlowInstance<GraphFlowNode, GraphFlowEdge>

// The subset of a node the attachment pass needs: its absolute position and
// whatever React Flow has measured so far.
interface RoutableNode {
  id: string
  position: { x: number, y: number }
  measured?: { width?: number, height?: number }
}

export interface ArchitectureDiagramSurfaceProps {
  graph: DiagramGraph
  onChange: (graph: DiagramGraph) => void
  className?: string
  readOnly?: boolean
  /** How connections are drawn: a window preference, LikeC4 has no such setting. */
  edgeStyle: EdgeStyle
  /** The user picked another style; the surface already dropped every saved bend, which meant something else in the old one. */
  onEdgeStyleChange: (style: EdgeStyle) => void
  /** Each time this turns true, the view refits the whole graph to the frame (e.g. when its host maximizes it). */
  autoFit?: boolean
  /** Zoom of the canvas embedding this graph (1 when standalone); the graph magnifies with it. */
  hostScale?: number
  /** Extra controls for the toolbar, e.g. what a view shows. */
  toolbar?: ReactNode
  /** Given (saved views), removing a node only takes it out of this view; otherwise it deletes the element from the model. */
  onRemoveFromView?: (nodeIds: string[]) => void
  /** Selects this node whenever it changes, e.g. the module a window was opened on. */
  select?: string | null
  /** The model's element kinds, offered on a selected node. */
  kinds?: readonly string[]
  /** Whether drawing outside every element creates one. A saved view has no element of its own to put it in. */
  topLevelDrawing?: boolean
}

interface ArchitectureNodeData extends DiagramNode, Record<string, unknown> {
  onChange: (id: string, patch: Partial<Pick<DiagramNode, "title" | "bullets" | "kind" | "width" | "height" | "position">>) => void
  /** Kinds a selected node may switch to; empty when it cannot. */
  kinds: readonly string[]
  /** True exactly once, for the node the user just created. */
  claimAutoEdit?: (id: string) => boolean
  /** The corner X: what it says and does; absent when read-only. */
  remove?: NodeRemoval
}

interface NodeRemoval {
  label: string
  run: (ids: string[], edges?: DiagramEdge[]) => void
}

/** Stable, so an unset `kinds` prop does not rebuild every node each render. */
const NO_KINDS: readonly string[] = []

// Rough.js is the same seeded, multi-stroke renderer Excalidraw uses. Stable
// seeds keep the drawing still across React renders instead of making it jitter.
const roughGenerator = rough.generator()

function roughSeed(id: string): number {
  let hash = 2166136261
  for (let index = 0; index < id.length; index += 1) hash = Math.imul(hash ^ id.charCodeAt(index), 16777619)
  return (hash >>> 0) || 1
}

// Curved connections draw as a single wobbly line: rough.js's doubled stroke
// reads as jitter at edge scale rather than pencil. Elbow connections stay crisp.
function edgeSketchPaths(d: string, seed: number, strokeWidth: number) {
  return roughGenerator.toPaths(roughGenerator.path(d, {
    seed, roughness: 1, bowing: 1, stroke: "currentColor", strokeWidth, preserveVertices: true, disableMultiStroke: true,
  }))
}

// A rounded rectangle, sketched by rough.js like Excalidraw's boxes. The fill
// is sketched too, so it follows the outline's wobble instead of sitting under
// it as a perfect CSS shape. rough.js only echoes `fill` back on the fill path;
// the actual colour comes from CSS so it tracks the theme.
function nodeSketchPaths(width: number, height: number, seed: number, strokeWidth: number) {
  const options = { seed, roughness: 1, bowing: 1, stroke: "currentColor", strokeWidth, preserveVertices: true, fill: "sketch", fillStyle: "solid" }
  const radius = Math.max(0, Math.min(9, (width - 2) / 2, (height - 2) / 2))
  const d = `M ${radius} 1 H ${width - radius} Q ${width - 1} 1 ${width - 1} ${radius} V ${height - radius} Q ${width - 1} ${height - 1} ${width - radius} ${height - 1} H ${radius} Q 1 ${height - 1} 1 ${height - radius} V ${radius} Q 1 1 ${radius} 1 Z`
  return roughGenerator.toPaths(roughGenerator.path(d, options))
}

// Any closed loop drawn on the canvas becomes a node in its bounding box; an
// open stroke (a line, a tick) is ignored. Every element is drawn as the same
// box, so the loop's shape carries no meaning beyond its extent.
const MIN_DRAW_SIZE = 30
const MAX_OPEN_GAP_RATIO = 0.3

/**
 * The box a freehand loop encloses, in the points' own coordinate space (the
 * surface passes flow coordinates). Null for fewer than three points, an open
 * stroke (ends further apart than MAX_OPEN_GAP_RATIO of its length), or a loop
 * under MIN_DRAW_SIZE on both axes.
 */
export function drawnBox(points: Array<{ x: number, y: number }>): { x: number, y: number, width: number, height: number } | null {
  if (points.length < 3) return null
  const xs = points.map((p) => p.x)
  const ys = points.map((p) => p.y)
  const minX = Math.min(...xs), maxX = Math.max(...xs)
  const minY = Math.min(...ys), maxY = Math.max(...ys)
  const width = maxX - minX
  const height = maxY - minY
  if (Math.max(width, height) < MIN_DRAW_SIZE) return null

  let pathLength = 0
  for (let index = 1; index < points.length; index += 1) pathLength += Math.hypot(points[index].x - points[index - 1].x, points[index].y - points[index - 1].y)
  const gap = Math.hypot(points[points.length - 1].x - points[0].x, points[points.length - 1].y - points[0].y)
  if (pathLength === 0 || gap / pathLength > MAX_OPEN_GAP_RATIO) return null
  return { x: minX, y: minY, width, height }
}

// Structural styling stays inline: React Flow must measure a real box even if
// the stylesheet chunk has not loaded yet. The visible border is an SVG sketch.
const cardStyle: CSSProperties = {
  width: "100%",
  height: "100%",
  position: "relative",
  overflow: "visible",
  border: 0,
  backgroundColor: "transparent",
}

// Minimum on-screen size a drawn stroke's bounding box must reach to register,
// independent of canvas zoom.
const DRAW_MIN_SCREEN_SIZE = 30
const isTap = (points: ReadonlyArray<{ x: number, y: number }>) => {
  const xs = points.map((p) => p.x)
  const ys = points.map((p) => p.y)
  return points.length === 0 || Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) < DRAW_MIN_SCREEN_SIZE
}
// New nodes spawn at a 3:2 width:height ratio.
const NEW_NODE_WIDTH = 240
const NEW_NODE_HEIGHT = 160
const NODE_MIN_HEIGHT = 48
// Drag/resize commits round to this grid, so autosaved positions stay stable
// pixel values instead of accumulating sub-pixel drift across edit sessions.
const GRID_SIZE = 8
const snapToGrid = (value: number) => Math.round(value / GRID_SIZE) * GRID_SIZE

// Clones `candidates` as titled drafts, so each copy becomes a new element,
// offset diagonally so the copies read as new objects rather than sitting
// exactly on their sources. Ids avoid `occupied` and each other.
// Compounds are skipped: copying one would not copy what it contains.
export function duplicateNodes(candidates: readonly DiagramNode[], occupied: Iterable<string>): DiagramNode[] {
  const taken = new Set(occupied)
  const clones: DiagramNode[] = []
  for (const node of candidates) {
    if (node.compound) continue
    const id = nextDraftId(taken)
    taken.add(id)
    clones.push({ ...node, id, position: { x: snapToGrid(node.position.x + GRID_SIZE * 3), y: snapToGrid(node.position.y + GRID_SIZE * 3) } })
  }
  return clones
}

interface PendingFocus {
  index: number
  cursor: number
}

/**
 * A card's text editor: the title and bullet drafts, moving the caret between
 * them, and one commit of both when focus leaves the card. Typing only touches
 * the drafts, so moving from the heading into the bullets never turns a new
 * node into an element mid-edit (that swaps its id and would drop the caret).
 */
function useCardText(data: ArchitectureNodeData, cardRef: RefObject<HTMLDivElement | null>) {
  const titleRef = useRef<HTMLInputElement>(null)
  const bulletRefs = useRef<Array<HTMLTextAreaElement | null>>([])
  const [editingTitle, setEditingTitle] = useState(false)

  // Model updates refresh the drafts only while nobody is typing in this card.
  const typing = () => !!cardRef.current?.contains(document.activeElement)
  const [titleDraft, setTitleDraft] = useState(data.title)
  useLayoutEffect(() => {
    if (typing()) return
    setTitleDraft(data.title)
  }, [data.title])
  const [bulletDrafts, setBulletDrafts] = useState<string[]>(() => [...data.bullets])
  useLayoutEffect(() => {
    if (typing()) return
    setBulletDrafts([...data.bullets])
  }, [data.bullets])
  // Drop refs for removed rows so the autosize pass cannot touch an unmounted textarea.
  bulletRefs.current.length = bulletDrafts.length
  const [pendingFocus, setPendingFocus] = useState<PendingFocus | null>(null)
  useLayoutEffect(() => {
    if (!pendingFocus) return
    const el = bulletRefs.current[pendingFocus.index]
    el?.focus()
    el?.setSelectionRange(pendingFocus.cursor, pendingFocus.cursor)
    setPendingFocus(null)
  }, [pendingFocus])
  const focusBullet = (index: number, cursor: number) => setPendingFocus({ index, cursor })

  // The commit reads the drafts through refs: a key handler may change them and
  // blur in the same event, before React re-renders.
  const titleDraftRef = useRef(titleDraft)
  titleDraftRef.current = titleDraft
  const bulletDraftsRef = useRef(bulletDrafts)
  bulletDraftsRef.current = bulletDrafts
  const setBullets = (next: string[]) => {
    bulletDraftsRef.current = next
    setBulletDrafts(next)
  }
  const handleCardBlur = (event: FocusEvent<HTMLDivElement>) => {
    if (event.currentTarget.contains(event.relatedTarget as globalThis.Node | null)) return
    const title = titleDraftRef.current
    const bullets = bulletDraftsRef.current.map((line) => line.replace(/^\s*[•-]\s?/, "").trim()).filter(Boolean)
    // Leaving drops empty lines locally too, even when nothing reaches the model.
    if (bullets.length !== bulletDraftsRef.current.length) setBullets(bullets)
    if (title === data.title && bullets.join("\n") === data.bullets.join("\n")) return
    data.onChange(data.id, { title, bullets })
  }

  // ArrowDown moves into the bullets; Enter does too, starting the first one if there is none.
  const handleTitleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "ArrowDown" && bulletDrafts.length > 0) {
      event.preventDefault()
      focusBullet(0, 0)
    } else if (event.key === "Enter") {
      event.preventDefault()
      if (bulletDrafts.length === 0) setBullets([""])
      focusBullet(0, 0)
    }
  }
  const handleBulletKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>, index: number) => {
    const textarea = event.currentTarget
    if (event.key === "Enter" && !event.shiftKey && !textarea.value.trim()) {
      // Enter on an empty bullet finishes the card: the line goes, focus leaves, the card commits.
      event.preventDefault()
      setBullets(bulletDrafts.filter((_, i) => i !== index))
      textarea.blur()
    } else if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault()
      const pos = textarea.selectionStart
      const next = [...bulletDrafts]
      next.splice(index, 1, textarea.value.slice(0, pos), textarea.value.slice(pos))
      setBullets(next)
      focusBullet(index + 1, 0)
    } else if (event.key === "Backspace" && index > 0 && textarea.selectionStart === 0 && textarea.selectionEnd === 0) {
      event.preventDefault()
      const mergeAt = bulletDrafts[index - 1].length
      const next = [...bulletDrafts]
      next.splice(index - 1, 2, next[index - 1] + next[index])
      setBullets(next)
      focusBullet(index - 1, mergeAt)
    } else if ((event.key === "ArrowUp" || event.key === "ArrowDown") && !event.shiftKey && !event.altKey && !event.metaKey && !event.ctrlKey) {
      // Inside a wrapped bullet the arrow moves the caret between visual lines;
      // only when the browser leaves the caret where it was (first/last line) do we hop rows.
      const before = textarea.selectionStart
      const target = index + (event.key === "ArrowUp" ? -1 : 1)
      requestAnimationFrame(() => {
        if (!textarea.isConnected || textarea.selectionStart !== before || textarea.selectionEnd !== before) return
        if (target < 0) { setEditingTitle(true); return }
        if (target >= bulletDrafts.length) return
        focusBullet(target, Math.min(before, bulletDrafts[target].length))
      })
    }
  }
  const editBullet = (index: number, text: string) => setBullets(bulletDrafts.map((line, i) => i === index ? text : line))
  // Starts a new bullet below the others, or reuses a trailing empty one.
  const startBullet = () => {
    const last = bulletDrafts.length - 1
    if (last >= 0 && !bulletDrafts[last].trim()) { focusBullet(last, 0); return }
    setBullets([...bulletDrafts, ""])
    focusBullet(bulletDrafts.length, 0)
  }

  useLayoutEffect(() => {
    if (data.claimAutoEdit?.(data.id)) setEditingTitle(true)
    // Mount only: a freshly created node claims its one-time edit, nothing else does.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  useLayoutEffect(() => {
    if (!editingTitle) return
    // React Flow keeps a new node visibility:hidden until it has measured it,
    // and a hidden input silently refuses focus, so retry for a few frames.
    let frame = 0
    let attempts = 0
    const focusTitle = () => {
      const input = titleRef.current
      if (!input || document.activeElement === input) return
      input.focus()
      if (document.activeElement !== input && attempts++ < 10) frame = requestAnimationFrame(focusTitle)
    }
    focusTitle()
    return () => cancelAnimationFrame(frame)
  }, [editingTitle])

  return { titleRef, bulletRefs, editingTitle, setEditingTitle, titleDraft, setTitleDraft, bulletDrafts, editBullet, startBullet, handleCardBlur, handleTitleKeyDown, handleBulletKeyDown }
}

function ArchitectureNodeCard({ data, selected }: NodeProps<GraphFlowNode>) {
  const [hovered, setHovered] = useState(false)
  const cardRef = useRef<HTMLDivElement>(null)
  const [cardSize, setCardSize] = useState(FALLBACK_SIZE)
  useLayoutEffect(() => {
    const card = cardRef.current
    if (!card) return
    const update = (width: number, height: number) => {
      const next = { width: Math.max(1, Math.round(width)), height: Math.max(1, Math.round(height)) }
      setCardSize((current) => current.width === next.width && current.height === next.height ? current : next)
    }
    update(card.offsetWidth, card.offsetHeight)
    if (typeof ResizeObserver === "undefined") return
    const observer = new ResizeObserver(([entry]) => {
      if (entry) update(entry.contentRect.width, entry.contentRect.height)
    })
    observer.observe(card)
    return () => observer.disconnect()
  }, [])
  const { titleRef, bulletRefs, editingTitle, setEditingTitle, titleDraft, setTitleDraft, bulletDrafts, editBullet, startBullet, handleCardBlur, handleTitleKeyDown, handleBulletKeyDown } = useCardText(data, cardRef)
  // A title bar wider than the card would truncate the heading and push the kind
  // label past the edge, so their width is a resize floor (and grows the card while typing).
  const minNodeWidth = nodeMinWidth(titleDraft, data.kind)
  // Auto-fit: the content's natural size is a floor for the card. The card
  // grows when text stops fitting (vertically or horizontally) and never
  // shrinks on its own, so a size the user chose sticks until the text needs more.
  const titlebarRef = useRef<HTMLDivElement>(null)
  const bodyRef = useRef<HTMLDivElement>(null)
  const resizingRef = useRef(false)
  const [fitHeight, setFitHeight] = useState(NODE_MIN_HEIGHT)
  useEffect(() => {
    const current = data.width ?? FALLBACK_SIZE.width
    if (minNodeWidth > current) data.onChange(data.id, { width: minNodeWidth })
  }, [minNodeWidth, data.width, data.id, data.onChange])
  const nodeSketch = useMemo(() => {
    const paths = nodeSketchPaths(cardSize.width, cardSize.height, roughSeed(data.id), selected ? 1.9 : hovered ? 1.7 : 1.5)
    return { fills: paths.filter((path) => path.fill !== "none"), strokes: paths.filter((path) => path.fill === "none") }
  }, [cardSize, data.id, hovered, selected])
  const controlsVisible = hovered || selected
  // Hidden at rest so a sketch stays a sketch. Still always interactive, so
  // pen/touch (no hover state) can tap a handle; a tap that selects the node
  // reveals them.
  const controlStyle = { opacity: controlsVisible ? 1 : 0, pointerEvents: "auto" as const }
  useEffect(() => {
    // scrollHeight forces a sync reflow; defer to next frame to avoid one per keystroke.
    const id = requestAnimationFrame(() => {
      for (const textarea of bulletRefs.current) {
        if (!textarea) continue
        textarea.style.height = "auto"
        textarea.style.height = `${textarea.scrollHeight}px`
      }
      const titlebar = titlebarRef.current
      const body = bodyRef.current
      if (!titlebar || !body) return
      // The body stretches to fill the card, so its own height says nothing
      // about its content: add the rows up instead.
      const style = getComputedStyle(body)
      const rows = Array.from(body.children, (row) => (row as HTMLElement).offsetHeight)
      const needed = titlebar.offsetHeight + parseFloat(style.paddingTop) + parseFloat(style.paddingBottom)
        + rows.reduce((sum, height) => sum + height, 0) + (parseFloat(style.rowGap) || 0) * Math.max(0, rows.length - 1)
      const fit = Math.max(NODE_MIN_HEIGHT, Math.ceil(needed / GRID_SIZE) * GRID_SIZE)
      setFitHeight(fit)
      if (resizingRef.current) return
      // A word wider than the text box can't wrap, so it clips; widen the card until it fits.
      const overflowX = bulletRefs.current.reduce((most, textarea) => textarea ? Math.max(most, textarea.scrollWidth - textarea.clientWidth) : most, 0)
      const width = data.width ?? FALLBACK_SIZE.width
      const nextWidth = overflowX > 0 ? Math.ceil((width + overflowX) / GRID_SIZE) * GRID_SIZE : width
      const growHeight = data.height !== undefined && fit > data.height
      if (!growHeight && nextWidth === width) return
      data.onChange(data.id, { ...(growHeight ? { height: fit } : {}), ...(nextWidth !== width ? { width: nextWidth } : {}) })
    })
    return () => cancelAnimationFrame(id)
  }, [bulletDrafts, titleDraft, editingTitle, cardSize, data.height, data.width, data.id, data.onChange, bulletRefs])
  return (
    <div ref={cardRef} className={cn("architecture-diagram-node", data.compound && "architecture-diagram-node-compound", selected && "architecture-diagram-node-selected")} style={cardStyle} onMouseEnter={() => setHovered(true)} onMouseLeave={() => setHovered(false)} onBlur={handleCardBlur}>
      <NodeResizer isVisible={!!selected} minWidth={minNodeWidth} minHeight={fitHeight} color="var(--sketch-ink-soft)" onResizeStart={() => { resizingRef.current = true }} onResizeEnd={(_event, params) => {
        resizingRef.current = false
        const x = snapToGrid(params.x)
        const y = snapToGrid(params.y)
        data.onChange(data.id, { width: snapToGrid(params.x + params.width) - x, height: snapToGrid(params.y + params.height) - y, position: { x, y } })
      }} />
      {/* Fill sits under the content (a later positioned sibling); the outline sits on top. */}
      <svg aria-hidden="true" focusable="false" style={{ position: "absolute", inset: 0, width: "100%", height: "100%", overflow: "visible", pointerEvents: "none" }}>
        {nodeSketch.fills.map((path, index) => <path key={index} d={path.d} className="architecture-diagram-node-fill" stroke="none" />)}
      </svg>
      <svg className="architecture-diagram-node-sketch" aria-hidden="true" focusable="false" style={{ position: "absolute", zIndex: 1, inset: 0, width: "100%", height: "100%", overflow: "visible", pointerEvents: "none" }}>
        {nodeSketch.strokes.map((path, index) => <path key={index} d={path.d} fill="none" stroke="currentColor" strokeWidth={path.strokeWidth} />)}
      </svg>
      <Handle id="top" type="source" position={Position.Top} className="architecture-diagram-handle" style={controlStyle} />
      {data.remove && controlsVisible && (
        <button type="button" aria-label={data.remove.label} className="nodrag nopan absolute right-1 top-1 rounded-full bg-surface-elevated p-0.5 text-ink-muted hover:text-danger" style={{ zIndex: 2 }}
          onPointerDown={(event) => event.stopPropagation()} onClick={(event) => { event.stopPropagation(); data.remove!.run([data.id]) }}>
          <X className="h-3 w-3" />
        </button>
      )}
      <Handle id="left" type="source" position={Position.Left} className="architecture-diagram-handle" style={controlStyle} />
      {/* Not clipped: the kind list opens past the card's edge. The body and title clip themselves.
          Inert until the card is activated (see globals.css), so a click selects it first. */}
      <div className="architecture-diagram-node-content" style={{ position: "relative", height: "100%", display: "grid" }}>
      <div style={{ display: "flex", flexDirection: "column", minWidth: 0, minHeight: 0, width: "100%", height: "100%" }}>
      <div ref={titlebarRef} className="architecture-diagram-node-titlebar">
        {editingTitle ? (
          <input ref={titleRef} autoFocus aria-label="Node title" value={titleDraft} onChange={(event) => setTitleDraft(event.target.value)} onBlur={() => setEditingTitle(false)} onKeyDown={handleTitleKeyDown} onPointerDown={(event) => event.stopPropagation()} className="nodrag architecture-diagram-title architecture-diagram-title-input" />
        ) : (
          <span role="button" tabIndex={0} aria-label="Edit node title" onClick={() => setEditingTitle(true)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); setEditingTitle(true) } }} className="architecture-diagram-title architecture-diagram-title-display">{titleDraft}</span>
        )}
        {data.kinds.length > 0 && !isDraft(data.id)
          // Looks exactly like the label; on an activated card, clicking it opens the list.
          ? <div className="nodrag nopan" style={{ flexShrink: 0 }} onPointerDown={(event) => event.stopPropagation()}>
            <StyledSelect variant="inline" className="architecture-diagram-node-kind" ariaLabel="Kind" value={data.kind} options={data.kinds.map((kind) => ({ value: kind, label: kind }))} onChange={(kind) => data.onChange(data.id, { kind })} />
          </div>
          : data.kind && <span className="architecture-diagram-node-kind">{data.kind}</span>}
      </div>
      {/* A compound's body is where its children sit, so its bullets (the element's
          description) stay hidden on the canvas; they remain editable in Text. */}
      {/* Clicking below the bullets starts a new one. Focusable, so that click keeps
          focus inside the card instead of committing it. */}
      <div ref={bodyRef} tabIndex={-1} className="architecture-diagram-node-body nowheel" style={{ outline: "none" }} hidden={data.compound} onClick={(event) => { if (event.target === event.currentTarget) startBullet() }}>
        {bulletDrafts.map((text, index) => (
          <div key={index} className="architecture-diagram-bullet-row">
            <span className="architecture-diagram-bullet-marker" aria-hidden="true">—</span>
            <textarea
              ref={(el) => { bulletRefs.current[index] = el }}
              aria-label="Node bullet"
              value={text}
              onChange={(event) => editBullet(index, event.target.value)}
              onKeyDown={(event) => handleBulletKeyDown(event, index)}
              className="nodrag architecture-diagram-bullet-input"
              rows={1}
              style={{ resize: "none" }}
            />
          </div>
        ))}
      </div>
      </div>
      </div>
      <Handle id="right" type="source" position={Position.Right} className="architecture-diagram-handle" style={controlStyle} />
      <Handle id="bottom" type="source" position={Position.Bottom} className="architecture-diagram-handle" style={controlStyle} />
    </div>
  )
}

// Floating edges: the attachment point is recomputed from the two nodes' live
// rectangles on every render, so dragging a node re-routes its edges to the
// nearest side instead of leaving a long way-around path behind.

// Pick the side of `node` facing `toward`, returning that side's midpoint. Used
// until the slot pass below has geometry for both cards.
function attach(node: InternalNode<GraphFlowNode>, toward: InternalNode<GraphFlowNode>): AttachPoint {
  const rect = (item: InternalNode<GraphFlowNode>): CardRect => ({
    ...item.internals.positionAbsolute,
    width: item.measured.width ?? FALLBACK_SIZE.width,
    height: item.measured.height ?? FALLBACK_SIZE.height,
  })
  const own = rect(node)
  return slotPoint(own, sideToward(own, rect(toward)), 0.5)
}

// Soft avoidance: connections sharing a card side get their own slot on it
// instead of all leaving from the midpoint, so their stepped paths run in
// separate lanes. Lines may still cross — nothing here is a router.
const SIDE_SPREAD = 0.68

interface AttachPoint {
  x: number
  y: number
  position: Position
}

export interface EdgeAttachment {
  source: AttachPoint
  target: AttachPoint
}

interface CardRect {
  x: number
  y: number
  width: number
  height: number
}

function slotPoint(rect: CardRect, side: Position, ratio: number): AttachPoint {
  const { x, y, width, height } = rect
  if (side === Position.Top) return { x: x + width * ratio, y, position: side }
  if (side === Position.Bottom) return { x: x + width * ratio, y: y + height, position: side }
  if (side === Position.Left) return { x, y: y + height * ratio, position: side }
  return { x: x + width, y: y + height * ratio, position: side }
}

function sideToward(rect: CardRect, toward: CardRect): Position {
  const dx = toward.x + toward.width / 2 - (rect.x + rect.width / 2)
  const dy = toward.y + toward.height / 2 - (rect.y + rect.height / 2)
  // Compare against the card's own aspect so wide cards still prefer left/right.
  if (Math.abs(dx) * rect.height > Math.abs(dy) * rect.width) return dx > 0 ? Position.Right : Position.Left
  return dy > 0 ? Position.Bottom : Position.Top
}

/**
 * Slot points for every edge, keyed by edge id. `nodes` carry absolute flow
 * positions; one React Flow has not measured yet counts at FALLBACK_SIZE. An
 * edge with an endpoint missing from `nodes` is omitted, and its path falls
 * back to `attach`.
 */
export function edgeAttachments(nodes: readonly RoutableNode[], edges: readonly DiagramEdge[]): Map<string, EdgeAttachment> {
  const rects = new Map<string, CardRect>(nodes.map((node) => [node.id, {
    x: node.position.x, y: node.position.y,
    width: node.measured?.width ?? FALLBACK_SIZE.width,
    height: node.measured?.height ?? FALLBACK_SIZE.height,
  }]))
  const ends = edges.flatMap((edge) => {
    const source = rects.get(edge.source)
    const target = rects.get(edge.target)
    if (!source || !target) return []
    return [
      { edgeId: edge.id, role: "source" as const, nodeId: edge.source, rect: source, side: sideToward(source, target), toward: target },
      { edgeId: edge.id, role: "target" as const, nodeId: edge.target, rect: target, side: sideToward(target, source), toward: source },
    ]
  })
  const groups = new Map<string, typeof ends>()
  for (const end of ends) {
    const key = `${end.nodeId}:${end.side}`
    groups.set(key, [...(groups.get(key) ?? []), end])
  }
  const points = new Map<string, AttachPoint>()
  for (const group of groups.values()) {
    // Order along the side by where the other card sits, so neighbouring lines
    // keep their relative order and do not cross just to reach their slot.
    const horizontal = group[0].side === Position.Top || group[0].side === Position.Bottom
    const sorted = [...group].sort((a, b) => (horizontal
      ? a.toward.x + a.toward.width / 2 - (b.toward.x + b.toward.width / 2)
      : a.toward.y + a.toward.height / 2 - (b.toward.y + b.toward.height / 2)))
    sorted.forEach((end, index) => {
      const share = (index + 1) / (sorted.length + 1)
      const ratio = 0.5 + (share - 0.5) * SIDE_SPREAD
      points.set(`${end.edgeId}:${end.role}`, slotPoint(end.rect, end.side, ratio))
    })
  }
  return new Map(edges.flatMap((edge) => {
    const source = points.get(`${edge.id}:source`)
    const target = points.get(`${edge.id}:target`)
    return source && target ? [[edge.id, { source, target }] as const] : []
  }))
}

// Orthogonal connector, like a flowchart: leave each card perpendicular to its
// side. Facing sides give a Z whose middle lane `bend` shifts along the axis;
// perpendicular sides give a single L. Corners are rounded.
const CORNER_RADIUS = 8

interface Point { x: number, y: number }

function orthogonalRoute(from: AttachPoint, to: AttachPoint, bend: number) {
  const horizontal = (side: Position) => side === Position.Left || side === Position.Right
  let axis: "x" | "y" | null = null
  let raw: Point[]
  if (horizontal(from.position) && horizontal(to.position)) {
    const lane = (from.x + to.x) / 2 + bend
    raw = [from, { x: lane, y: from.y }, { x: lane, y: to.y }, to]
    axis = "x"
  } else if (!horizontal(from.position) && !horizontal(to.position)) {
    const lane = (from.y + to.y) / 2 + bend
    raw = [from, { x: from.x, y: lane }, { x: to.x, y: lane }, to]
    axis = "y"
  } else {
    raw = [from, horizontal(from.position) ? { x: to.x, y: from.y } : { x: from.x, y: to.y }, to]
  }
  const points = raw.filter((point, index) => index === 0 || Math.hypot(point.x - raw[index - 1].x, point.y - raw[index - 1].y) > 0.01)
  let d = `M ${points[0].x} ${points[0].y}`
  for (let index = 1; index < points.length - 1; index += 1) {
    const prev = points[index - 1]
    const corner = points[index]
    const next = points[index + 1]
    const before = Math.hypot(corner.x - prev.x, corner.y - prev.y)
    const after = Math.hypot(next.x - corner.x, next.y - corner.y)
    const radius = Math.min(CORNER_RADIUS, before / 2, after / 2)
    d += ` L ${corner.x - (corner.x - prev.x) / before * radius} ${corner.y - (corner.y - prev.y) / before * radius}`
      + ` Q ${corner.x} ${corner.y} ${corner.x + (next.x - corner.x) / after * radius} ${corner.y + (next.y - corner.y) / after * radius}`
  }
  const last = points[points.length - 1]
  d += ` L ${last.x} ${last.y}`
  // The lane is only draggable on a Z; the handle rides its middle segment.
  const draggable = axis !== null && points.length === 4
  const handle = draggable ? { x: (points[1].x + points[2].x) / 2, y: (points[1].y + points[2].y) / 2 } : null
  return { points, d, axis: draggable ? axis : null, handle }
}

function arrowHeadPath(tip: Point, from: Point): string {
  const angle = Math.atan2(tip.y - from.y, tip.x - from.x)
  const size = 10
  const wing = 0.4
  const left = { x: tip.x - Math.cos(angle - wing) * size, y: tip.y - Math.sin(angle - wing) * size }
  const right = { x: tip.x - Math.cos(angle + wing) * size, y: tip.y - Math.sin(angle + wing) * size }
  return `M ${left.x} ${left.y} L ${tip.x} ${tip.y} L ${right.x} ${right.y} Z`
}

function ArchitectureEdgePath({ id, source, target, data, selected }: EdgeProps<GraphFlowEdge>) {
  const sourceNode = useInternalNode<GraphFlowNode>(source)
  const targetNode = useInternalNode<GraphFlowNode>(target)
  const [dragBend, setDragBend] = useState<number | null>(null)
  const stopDragRef = useRef<(() => void) | null>(null)
  useEffect(() => () => stopDragRef.current?.(), [])
  const [editingLabel, setEditingLabel] = useState(false)
  useEffect(() => { if (!selected) setEditingLabel(false) }, [selected])
  // The label reaches the model once, when editing ends, not on every keystroke.
  const [labelDraft, setLabelDraft] = useState("")
  const commitLabel = () => {
    setEditingLabel(false)
    if (labelDraft.trim() !== (data?.label ?? "")) data?.onLabelChange?.(id, labelDraft.trim())
  }

  // Curved: `bend` is relative to the chord (endpoint-to-endpoint line), not an
  // absolute point. Elbow: it is the Z's middle-lane offset along its axis. Both
  // stay correct as nodes drag.
  const elbow = data?.edgeStyle === "elbow"
  const geometry = useMemo(() => {
    if (!sourceNode || !targetNode) return null
    const from = data?.attachment?.source ?? attach(sourceNode, targetNode)
    const to = data?.attachment?.target ?? attach(targetNode, sourceNode)
    const bend = dragBend ?? data?.path?.bend ?? 0
    if (elbow) {
      const route = orthogonalRoute(from, to, bend)
      const { points } = route
      const laneBase = route.axis ? (from[route.axis] + to[route.axis]) / 2 : 0
      const axis = route.axis
      return {
        rough: false,
        edgePath: route.d,
        arrows: arrowHeadPath(points[points.length - 1], points[points.length - 2] ?? from),
        handle: route.handle,
        labelPoint: route.handle ?? points[Math.floor(points.length / 2)],
        project: (point: Point) => axis ? point[axis] - laneBase : null,
        nudge: (key: string) => axis === "x" ? (key === "ArrowRight" ? 1 : key === "ArrowLeft" ? -1 : null)
          : axis === "y" ? (key === "ArrowDown" ? 1 : key === "ArrowUp" ? -1 : null) : null,
        keys: axis === "x" ? "ArrowLeft ArrowRight" : "ArrowUp ArrowDown",
      }
    }
    const mid = { x: (from.x + to.x) / 2, y: (from.y + to.y) / 2 }
    const length = Math.hypot(to.x - from.x, to.y - from.y) || 1
    const normal = { x: -(to.y - from.y) / length, y: (to.x - from.x) / length }
    // A quadratic Bézier does not pass through its control point, so the
    // control sits twice as far out as the point the user actually sees.
    const pathPoint = { x: mid.x + normal.x * bend, y: mid.y + normal.y * bend }
    const control = { x: mid.x + normal.x * bend * 2, y: mid.y + normal.y * bend * 2 }
    return {
      rough: true,
      edgePath: `M ${from.x} ${from.y} Q ${control.x} ${control.y} ${to.x} ${to.y}`,
      arrows: arrowHeadPath(to, control),
      handle: pathPoint,
      labelPoint: pathPoint,
      project: (point: Point) => (point.x - mid.x) * normal.x + (point.y - mid.y) * normal.y,
      nudge: (key: string) => key === "ArrowUp" || key === "ArrowRight" ? 1 : key === "ArrowDown" || key === "ArrowLeft" ? -1 : null,
      keys: "ArrowUp ArrowDown ArrowLeft ArrowRight",
    }
  }, [sourceNode, targetNode, elbow, data?.attachment, data?.path?.bend, dragBend])

  const sketch = useMemo(
    () => geometry?.rough ? edgeSketchPaths(geometry.edgePath, roughSeed(id), selected ? 2 : 1.7) : [],
    [geometry, id, selected],
  )

  if (!geometry) return null
  const { labelPoint, handle } = geometry

  const startDrag = (event: ReactPointerEvent<SVGCircleElement>) => {
    event.preventDefault()
    event.stopPropagation()
    const { project } = geometry
    const move = (pointer: PointerEvent) => {
      const point = data?.toFlowPoint?.(pointer.clientX, pointer.clientY)
      if (!point) return
      const value = project(point)
      if (value !== null) setDragBend(value)
    }
    const stop = () => {
      window.removeEventListener("pointermove", move)
      window.removeEventListener("pointerup", stop)
      window.removeEventListener("pointercancel", stop)
      stopDragRef.current = null
      setDragBend((current) => {
        if (current !== null) data?.onPathChange?.(id, { bend: current })
        return null
      })
    }
    stopDragRef.current = stop
    window.addEventListener("pointermove", move)
    window.addEventListener("pointerup", stop, { once: true })
    window.addEventListener("pointercancel", stop, { once: true })
  }
  const nudge = (event: KeyboardEvent<SVGCircleElement>) => {
    const step = event.shiftKey ? 10 : 1
    const sign = geometry.nudge(event.key)
    if (sign === null) return
    event.preventDefault()
    data?.onPathChange?.(id, { bend: (data?.path?.bend ?? 0) + sign * step })
  }

  return <>
    <g onDoubleClick={() => { if (data?.onLabelChange) { setLabelDraft(data.label ?? ""); setEditingLabel(true) } }}>
      <BaseEdge id={id} path={geometry.edgePath} interactionWidth={20} style={{ stroke: "transparent" }} />
    </g>
    {geometry.rough
      ? sketch.map((path, index) => <path key={index} d={path.d} fill="none" stroke="currentColor" strokeWidth={path.strokeWidth} strokeLinecap="round" className={cn("architecture-diagram-edge", selected && "architecture-diagram-edge-selected")} pointerEvents="none" />)
      : <path d={geometry.edgePath} fill="none" stroke="currentColor" strokeWidth={selected ? 2 : 1.6} strokeLinecap="round" strokeLinejoin="round" className={cn("architecture-diagram-edge", selected && "architecture-diagram-edge-selected")} pointerEvents="none" />}
    <path d={geometry.arrows} fill="currentColor" stroke="currentColor" strokeWidth={1.2} strokeLinejoin="round" className={cn("architecture-diagram-edge", selected && "architecture-diagram-edge-selected")} pointerEvents="none" />
    {selected && editingLabel && data?.onLabelChange
      ? <foreignObject x={labelPoint.x - 70} y={labelPoint.y - 26} width={140} height={24} overflow="visible"><input autoFocus aria-label="Connection label" value={labelDraft} placeholder="Label" onChange={(event) => setLabelDraft(event.target.value)} onBlur={commitLabel} onKeyDown={(event) => { if (event.key === "Enter") { event.preventDefault(); event.currentTarget.blur() } else if (event.key === "Escape") { event.preventDefault(); setEditingLabel(false) } }} className="architecture-diagram-edge-label-input nodrag nopan nowheel" /></foreignObject>
      : data?.label && <text x={labelPoint.x} y={labelPoint.y - 12} className="architecture-diagram-edge-label" textAnchor="middle" dominantBaseline="central">{data.label}</text>}
    {selected && handle && data?.onPathChange && <circle cx={handle.x} cy={handle.y} r={6} role="button" tabIndex={0} aria-label="Adjust connection curve" aria-keyshortcuts={geometry.keys} className="architecture-diagram-path-handle nodrag nopan" onPointerDown={startDrag} onKeyDown={nudge} />}
  </>
}

const nodeTypes = { "architecture-node": ArchitectureNodeCard }
const edgeTypes = { "architecture-edge": ArchitectureEdgePath }

// React Flow wants a parent before its children, and each child positioned
// relative to its parent; the graph keeps absolute positions throughout.
function toFlowNodes(nodes: DiagramNode[], onNodeChange: ArchitectureNodeData["onChange"], kinds: readonly string[], claimAutoEdit?: ArchitectureNodeData["claimAutoEdit"], remove?: NodeRemoval): GraphFlowNode[] {
  const byId = new Map(nodes.map((node) => [node.id, node]))
  const depth = (node: DiagramNode): number => {
    const parent = node.parent ? byId.get(node.parent) : undefined
    return parent ? 1 + depth(parent) : 0
  }
  return [...nodes].sort((a, b) => depth(a) - depth(b)).map((node) => {
    const parent = node.parent ? byId.get(node.parent) : undefined
    return {
      id: node.id, type: "architecture-node",
      position: parent ? { x: node.position.x - parent.position.x, y: node.position.y - parent.position.y } : node.position,
      ...(parent ? { parentId: parent.id } : {}),
      // Height is left unset for nodes without an explicit height, so they grow with their content.
      style: { width: node.width ?? FALLBACK_SIZE.width, ...(node.height !== undefined ? { height: node.height } : {}) },
      data: { ...node, onChange: onNodeChange, kinds, claimAutoEdit, remove },
    }
  })
}

// Live flow nodes in absolute coordinates, so routing works across nesting levels mid-drag.
function absoluteNodes(nodes: readonly GraphFlowNode[]): RoutableNode[] {
  const byId = new Map(nodes.map((node) => [node.id, node]))
  const absolute = (node: GraphFlowNode): { x: number, y: number } => {
    const parent = node.parentId ? byId.get(node.parentId) : undefined
    if (!parent) return node.position
    const origin = absolute(parent)
    return { x: origin.x + node.position.x, y: origin.y + node.position.y }
  }
  return nodes.map((node) => ({ id: node.id, position: absolute(node), measured: node.measured }))
}

/**
 * The deepest node whose box holds the point: where a drawn or dropped node
 * belongs. Drawing may nest into any element (a leaf then becomes a parent),
 * dropping only into one that already frames others. Ids are element FQNs, so
 * nesting depth is the FQN's length, and `excluded` drops a node's own subtree.
 */
function innermostNode(nodes: readonly DiagramNode[], x: number, y: number, excluded: ReadonlySet<string>, compoundsOnly: boolean): string | null {
  let best: string | null = null
  for (const node of nodes) {
    if ((compoundsOnly && !node.compound) || isDraft(node.id) || [...excluded].some((id) => node.id === id || node.id.startsWith(`${id}.`))) continue
    const width = node.width ?? FALLBACK_SIZE.width
    const height = node.height ?? FALLBACK_SIZE.height
    if (x < node.position.x || y < node.position.y || x > node.position.x + width || y > node.position.y + height) continue
    if (best === null || node.id.split(".").length > best.split(".").length) best = node.id
  }
  return best
}

function toFlowEdges(edges: DiagramEdge[], edgeStyle: EdgeStyle, attachments = new Map<string, EdgeAttachment>(), runtime: EdgeRuntime = {}): GraphFlowEdge[] {
  return edges.map((edge) => ({
    ...edge,
    type: "architecture-edge",
    data: { ...edge, ...runtime, edgeStyle, ...(attachments.has(edge.id) ? { attachment: attachments.get(edge.id) } : {}) },
  }))
}

// Fields React Flow owns on a node/edge; the graph never describes them.
const RETAINED_BY_FLOW = ["selected", "measured", "dragging"] as const

// Rebuild flow items from the graph, carrying over only what React Flow owns. A
// blind replace clears selection and drops measurements on every edit, making
// edges jump and handles fade mid-interaction. Merging the other way round is
// just as wrong: it would keep keys the rebuilt item deliberately omits, e.g. a
// stale `data.attachment` after the endpoint node moves to a different side.
function reconcile<T extends { id: string }>(current: T[], next: T[]): T[] {
  const previous = new Map(current.map((item) => [item.id, item as Record<string, unknown>]))
  return next.map((item) => {
    const existing = previous.get(item.id)
    if (!existing) return item
    // An update landing mid-drag must not yank the node back to its stored position.
    if (existing.dragging) return existing as T
    const merged: Record<string, unknown> = { ...item }
    for (const key of RETAINED_BY_FLOW) if (key in existing) merged[key] = existing[key]
    return merged as T
  })
}

/**
 * The surface's camera: fitting, wheel and pinch zoom, and keeping the view
 * steady while the frame resizes or the host canvas zooms.
 */
function useSurfaceCamera({ flow, viewportRef, onNodesChange, autoFit, hostScale, readOnly }: {
  flow: GraphFlowInstance | null
  viewportRef: RefObject<HTMLDivElement | null>
  onNodesChange: (changes: NodeChange<GraphFlowNode>[]) => void
  autoFit: boolean
  hostScale: number
  readOnly: boolean
}) {
  const fitView = useCallback(() => flow?.fitView({ padding: 0.22, duration: 180 }), [flow])
  // Nodes measure and auto-grow after the first fit, which would leave the
  // graph cropped. Until the user takes the camera, keep refitting as sizes settle.
  const followFitRef = useRef(true)
  const followTimerRef = useRef<number | undefined>(undefined)
  useEffect(() => () => window.clearTimeout(followTimerRef.current), [])
  const takeCamera = useCallback(() => { followFitRef.current = false }, [])
  const handleNodesChange = useCallback((changes: NodeChange<GraphFlowNode>[]) => {
    onNodesChange(changes)
    if (!followFitRef.current || !changes.some((change) => change.type === "dimensions")) return
    window.clearTimeout(followTimerRef.current)
    followTimerRef.current = window.setTimeout(() => { if (followFitRef.current) void flow?.fitView({ padding: 0.22 }) }, 120)
  }, [onNodesChange, flow])
  useEffect(() => {
    if (!autoFit) return
    followFitRef.current = true
    // Two frames: the host's new size has to be laid out and picked up by React
    // Flow's own size observer before a fit can use it.
    let frame = requestAnimationFrame(() => { frame = requestAnimationFrame(() => { void fitView() }) })
    return () => cancelAnimationFrame(frame)
    // Refit on the transition to true only, not whenever the flow instance changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoFit])
  const zoomGraphAtCenter = useCallback((event: ReactWheelEvent<HTMLDivElement>) => {
    if (!flow || event.deltaY === 0) return
    takeCamera()
    event.preventDefault()
    event.stopPropagation()
    void (event.deltaY < 0 ? flow.zoomIn() : flow.zoomOut())
  }, [flow, takeCamera])
  // Two-finger pinch + pan: the same hook every canvas uses.
  useTwoFingerGesture(viewportRef, {
    enabled: !readOnly && !!flow,
    begin: () => { takeCamera(); return flow!.getViewport() },
    update: (start, { ratio, cx, cy, mx, my }) => {
      const zoom = Math.min(2 * hostScale, Math.max(0.2 * hostScale, start.zoom * ratio))
      flow!.setViewport({ x: mx - (cx - start.x) / start.zoom * zoom, y: my - (cy - start.y) / start.zoom * zoom, zoom })
    },
  })
  // React Flow pins its viewport to the top-left corner, so when the frame
  // resizes (fullscreen, a resize drag) the view's middle would drift. Both
  // this and host zoom re-project from the size the current viewport was set
  // for, keeping whatever sat in the center before in the center after.
  const viewportSizeRef = useRef({ width: 0, height: 0 })
  const hostScaleRef = useRef(hostScale)
  const reframe = useCallback((factor: number) => {
    const el = viewportRef.current
    if (!el || !flow) return
    const next = { width: el.clientWidth, height: el.clientHeight }
    const { x, y, zoom } = flow.getViewport()
    const view = reframeCamera({ x, y }, zoom, viewportSizeRef.current, next, factor)
    viewportSizeRef.current = next
    void flow.setViewport({ ...view.offset, zoom: view.scale })
  }, [flow, viewportRef])
  useEffect(() => {
    const el = viewportRef.current
    if (!el || !flow || typeof ResizeObserver === "undefined") return
    viewportSizeRef.current = { width: el.clientWidth, height: el.clientHeight }
    const observer = new ResizeObserver(() => {
      const { width, height } = viewportSizeRef.current
      if (el.clientWidth !== width || el.clientHeight !== height) reframe(1)
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [flow, reframe, viewportRef])
  // Embedded in a canvas, the frame grows and shrinks with the host's zoom;
  // magnify the graph by the same factor so the frame shows the same view,
  // just bigger or smaller. Runs before the ResizeObserver sees the new size.
  useLayoutEffect(() => {
    const factor = hostScale / hostScaleRef.current
    hostScaleRef.current = hostScale
    if (factor !== 1) reframe(factor)
  }, [hostScale, reframe])
  return { fitView, takeCamera, handleNodesChange, zoomGraphAtCenter }
}

/**
 * Pen drawing and the lasso. Drawing needs no tool: a closed pen stroke becomes
 * a node (`addDraft`). The lasso is the one opt-in tool: its loop selects the
 * nodes whose centres it encloses. Points are sampled in screen pixels and
 * converted to flow space when the stroke ends.
 */
function useSurfaceDrawing({ flow, viewportRef, readOnly, nodes, setNodes, setEdges, addDraft }: {
  flow: GraphFlowInstance | null
  viewportRef: RefObject<HTMLDivElement | null>
  readOnly: boolean
  nodes: GraphFlowNode[]
  setNodes: Dispatch<SetStateAction<GraphFlowNode[]>>
  setEdges: Dispatch<SetStateAction<GraphFlowEdge[]>>
  addDraft: (box: CardRect) => void
}) {
  const [lasso, setLasso] = useState(false)
  useEffect(() => {
    if (!lasso) return
    const onKeyDown = (event: globalThis.KeyboardEvent) => { if (event.key === "Escape") setLasso(false) }
    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [lasso])
  // Points accumulate in a ref so every pointer sample doesn't re-render the
  // whole surface; the preview only syncs to state once per animation frame.
  const drawPointsRef = useRef<Array<{ x: number, y: number }>>([])
  const drawFrameRef = useRef<number | null>(null)
  const [drawPreview, setDrawPreview] = useState<Array<{ x: number, y: number }> | null>(null)
  const drawOriginRef = useRef({ left: 0, top: 0 })
  useEffect(() => () => { if (drawFrameRef.current !== null) cancelAnimationFrame(drawFrameRef.current) }, [])
  const startDraw = useCallback((point: { clientX: number, clientY: number }) => {
    const rect = viewportRef.current?.getBoundingClientRect()
    if (!rect) return
    drawOriginRef.current = { left: rect.left, top: rect.top }
    drawPointsRef.current = [{ x: point.clientX, y: point.clientY }]
    setDrawPreview(drawPointsRef.current)
  }, [viewportRef])
  const onDrawPointerMove = useCallback((event: { clientX: number, clientY: number }) => {
    if (drawPointsRef.current.length === 0) return
    drawPointsRef.current.push({ x: event.clientX, y: event.clientY })
    if (drawFrameRef.current !== null) return
    drawFrameRef.current = requestAnimationFrame(() => {
      drawFrameRef.current = null
      setDrawPreview([...drawPointsRef.current])
    })
  }, [])
  // Whether a gesture was a tap is judged in screen pixels, before the points
  // are converted to flow space, so it does not change with zoom.
  const finishDraw = useCallback((commit: boolean) => {
    if (drawFrameRef.current !== null) { cancelAnimationFrame(drawFrameRef.current); drawFrameRef.current = null }
    const points = drawPointsRef.current
    drawPointsRef.current = []
    setDrawPreview(null)
    if (!commit || !flow || points.length === 0) return
    const tap = isTap(points)
    if (lasso) {
      // A tap clears the selection; a loop selects every node whose centre it
      // encloses, plus the connections between them, so one Delete removes the lot.
      const polygon = tap ? [] : points.map((p): [number, number] => { const at = flow.screenToFlowPosition(p); return [at.x, at.y] })
      const picked = new Set(absoluteNodes(nodes).filter((node) => pointInPolygon(
        node.position.x + (node.measured?.width ?? FALLBACK_SIZE.width) / 2,
        node.position.y + (node.measured?.height ?? FALLBACK_SIZE.height) / 2,
        polygon,
      )).map((node) => node.id))
      setNodes((current) => current.map((node) => ({ ...node, selected: picked.has(node.id) })))
      setEdges((current) => current.map((edge) => ({ ...edge, selected: picked.has(edge.source) && picked.has(edge.target) })))
      // Leave the tool once something is caught: the overlay would otherwise
      // swallow the drag that moves the selection.
      if (picked.size > 0) setLasso(false)
      return
    }
    // A tap is not a stroke: the pane click handler deselects.
    if (tap) return
    const box = drawnBox(points.map((p) => flow.screenToFlowPosition(p)))
    if (box) addDraft({ x: box.x, y: box.y, width: Math.max(NODE_MIN_WIDTH, Math.round(box.width)), height: Math.max(NODE_MIN_HEIGHT, Math.round(box.height)) })
  }, [flow, addDraft, lasso, nodes, setNodes, setEdges])
  // Pen draws, on the bare canvas and on any card that is not activated, which
  // is just more paper; touch and mouse pan. Window listeners, not pointer
  // capture: the pane still receives its click.
  const onPanePointerDown = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    if (readOnly || lasso || event.pointerType !== "pen" || event.button !== 0) return
    const target = event.target as Element
    const card = target.closest(".react-flow__node")
    const onCard = !!card && !card.classList.contains("selected") && !target.closest(".nodrag")
    if (!onCard && !target.classList.contains("react-flow__pane")) return
    // Stops the pen's compatibility mousedown, which would also start React Flow's pan.
    event.preventDefault()
    startDraw(event)
    const cleanup = () => {
      window.removeEventListener("pointermove", onDrawPointerMove)
      window.removeEventListener("pointerup", up)
      window.removeEventListener("pointercancel", cancel)
    }
    const up = () => {
      cleanup()
      const stroke = !isTap(drawPointsRef.current)
      finishDraw(true)
      if (!onCard || !stroke) return
      // The stroke's closing click would otherwise activate the card it was drawn on.
      const swallow = (click: MouseEvent) => click.stopPropagation()
      window.addEventListener("click", swallow, { capture: true, once: true })
      setTimeout(() => window.removeEventListener("click", swallow, { capture: true }))
    }
    const cancel = () => { cleanup(); finishDraw(false) }
    window.addEventListener("pointermove", onDrawPointerMove)
    window.addEventListener("pointerup", up)
    window.addEventListener("pointercancel", cancel)
  }, [readOnly, lasso, startDraw, onDrawPointerMove, finishDraw])
  // The lasso draws on an overlay that captures the pointer.
  const lassoOverlay = {
    onPointerDown: (event: ReactPointerEvent<HTMLDivElement>) => {
      event.currentTarget.setPointerCapture(event.pointerId)
      startDraw(event)
    },
    onPointerMove: onDrawPointerMove,
    onPointerUp: () => finishDraw(true),
    onPointerCancel: () => finishDraw(false),
  }
  const { left, top } = drawOriginRef.current
  const preview = drawPreview && drawPreview.length > 1 ? `M ${drawPreview.map((p) => `${p.x - left} ${p.y - top}`).join(" L ")}` : null
  return { lasso, setLasso, preview, onPanePointerDown, lassoOverlay }
}

export function ArchitectureDiagramSurface({ graph, onChange, className, readOnly = false, edgeStyle, onEdgeStyleChange, autoFit = false, hostScale = 1, toolbar, onRemoveFromView, select = null, kinds = NO_KINDS, topLevelDrawing = true }: ArchitectureDiagramSurfaceProps) {
  const active = useTabActive()
  const graphRef = useRef(graph)
  graphRef.current = graph
  const emit = useCallback((patch: Partial<DiagramGraph>) => onChange({ ...graphRef.current, ...patch }), [onChange])
  // Cards report positions in React Flow's frame, which is relative to the parent for nested nodes.
  const changeNode = useCallback((id: string, patch: Partial<Pick<DiagramNode, "title" | "bullets" | "kind" | "width" | "height" | "position">>) => {
    const nodes = graphRef.current.nodes
    const node = nodes.find((candidate) => candidate.id === id)
    const parent = node?.parent ? nodes.find((candidate) => candidate.id === node.parent) : undefined
    const absolute = patch.position && parent
      ? { ...patch, position: { x: patch.position.x + parent.position.x, y: patch.position.y + parent.position.y } }
      : patch
    emit({ nodes: nodes.map((candidate) => candidate.id === id ? { ...candidate, ...absolute } : candidate) })
  }, [emit])
  // The node just created by the toolbar or a drawn shape opens straight into
  // title editing. Claimed once on mount, so a later remount of the same node
  // (tab switch, reload) doesn't grab focus again.
  const autoEditIdRef = useRef<string | null>(null)
  const claimAutoEdit = useCallback((id: string) => {
    if (autoEditIdRef.current !== id) return false
    autoEditIdRef.current = null
    return true
  }, [])
  // One path for the corner X and the Delete key. A draft never reached the model, so it just goes.
  const removal = useMemo<NodeRemoval | undefined>(() => readOnly ? undefined : {
    label: onRemoveFromView ? "Remove from view" : "Delete element",
    // `edges`: connections the same Delete key press takes, written in the same change.
    run: (ids, edges) => {
      const saved = ids.filter((id) => !isDraft(id))
      if (onRemoveFromView && saved.length > 0) onRemoveFromView(saved)
      const doomed = new Set(onRemoveFromView ? ids.filter(isDraft) : ids)
      if (doomed.size > 0 || edges) emit({ nodes: graphRef.current.nodes.filter((node) => !doomed.has(node.id)), ...(edges ? { edges } : {}) })
    },
  }, [readOnly, onRemoveFromView, emit])
  const [flow, setFlow] = useState<GraphFlowInstance | null>(null)
  const viewportRef = useRef<HTMLDivElement>(null)
  const [edgeMenuOpen, setEdgeMenuOpen] = useState(false)
  const edgeMenuRef = useRef<HTMLDivElement>(null)
  useClickOutside(edgeMenuRef, useCallback(() => setEdgeMenuOpen(false), []))
  const changeEdgePath = useCallback((id: string, path?: DiagramEdge["path"]) => {
    emit({ edges: graphRef.current.edges.map((edge) => edge.id === id ? { ...edge, path } : edge) })
  }, [emit])
  const changeEdgeLabel = useCallback((id: string, label: string) => {
    emit({ edges: graphRef.current.edges.map((edge) => edge.id === id ? { ...edge, label } : edge) })
  }, [emit])
  const toFlowPoint = useCallback((x: number, y: number) => flow?.screenToFlowPosition({ x, y }), [flow])
  const edgeRuntime = useMemo<EdgeRuntime>(() => readOnly ? {} : {
    onPathChange: changeEdgePath,
    onLabelChange: changeEdgeLabel,
    toFlowPoint,
  }, [changeEdgePath, changeEdgeLabel, readOnly, toFlowPoint])
  const nodeKinds = readOnly ? NO_KINDS : kinds
  const [nodes, setNodes, onNodesChange] = useNodesState<GraphFlowNode>(toFlowNodes(graph.nodes, changeNode, nodeKinds, readOnly ? undefined : claimAutoEdit, removal))
  const [edges, setEdges, onEdgesChange] = useEdgesState<GraphFlowEdge>(toFlowEdges(graph.edges, edgeStyle, new Map(), edgeRuntime))
  const { fitView, takeCamera, handleNodesChange, zoomGraphAtCenter } = useSurfaceCamera({ flow, viewportRef, onNodesChange, autoFit, hostScale, readOnly })

  useEffect(() => {
    setNodes((current) => reconcile(current, toFlowNodes(graph.nodes, changeNode, nodeKinds, readOnly ? undefined : claimAutoEdit, removal)))
  }, [graph.nodes, changeNode, nodeKinds, claimAutoEdit, readOnly, removal, setNodes])
  useEffect(() => {
    if (select) setNodes((current) => current.map((node) => ({ ...node, selected: node.id === select })))
  }, [select, setNodes])
  // Only an activated (selected) card moves. Any other card is paper: a drag
  // on it pans (React Flow pans from non-draggable nodes), a pen draws on it,
  // and a click activates it.
  const flowNodes = useMemo(() => nodes.map((node) => node.draggable === !!node.selected ? node : { ...node, draggable: !!node.selected }), [nodes])
  // Recomputed from the live node rects, so dragging a card re-slots its edges.
  const attachments = useMemo(() => edgeAttachments(absoluteNodes(nodes), graph.edges), [nodes, graph.edges])
  useEffect(() => {
    setEdges((current) => reconcile(current, toFlowEdges(graph.edges, edgeStyle, attachments, edgeRuntime)))
  }, [graph.edges, edgeStyle, attachments, edgeRuntime, setEdges])

  // A new node lands in the innermost element under its centre, so drawing
  // inside an element creates something inside it. Where nothing frames it, a
  // view without its own canvas element (a saved view) takes nothing.
  const addDraft = useCallback((box: CardRect) => {
    const nodes = graphRef.current.nodes
    const id = nextDraftId(nodes.map((node) => node.id))
    const parent = innermostNode(nodes, box.x + box.width / 2, box.y + box.height / 2, new Set(), false)
    if (parent === null && !topLevelDrawing) return
    autoEditIdRef.current = id
    emit({ nodes: [...nodes, { id, title: "", bullets: [], kind: "", parent, compound: false, position: { x: box.x, y: box.y }, width: box.width, height: box.height }] })
  }, [emit, topLevelDrawing])
  const addNode = useCallback(() => {
    const offset = 100 + graphRef.current.nodes.length * 28
    addDraft({ x: offset, y: offset, width: NEW_NODE_WIDTH, height: NEW_NODE_HEIGHT })
  }, [addDraft])
  const { lasso, setLasso, preview, onPanePointerDown, lassoOverlay } = useSurfaceDrawing({ flow, viewportRef, readOnly, nodes, setNodes, setEdges, addDraft })
  // Adds clones to the graph and selects them, not their sources, so repeated
  // Ctrl+D / Ctrl+V walks diagonally instead of stacking on the same spot.
  const addClones = useCallback((clones: DiagramNode[]) => {
    if (clones.length === 0) return
    const cloneIds = new Set(clones.map((clone) => clone.id))
    emit({ nodes: [...graphRef.current.nodes, ...clones] })
    setNodes((current) => current.map((node) => ({ ...node, selected: cloneIds.has(node.id) })))
  }, [emit, setNodes])
  // The selected nodes as the graph has them.
  const selectedNodes = useCallback(() => {
    const ids = new Set(nodes.filter((node) => node.selected).map((node) => node.id))
    return graphRef.current.nodes.filter((node) => ids.has(node.id))
  }, [nodes])
  const duplicateSelectedNodes = useCallback(() => {
    addClones(duplicateNodes(selectedNodes(), graphRef.current.nodes.map((node) => node.id)))
  }, [selectedNodes, addClones])
  // A snapshot, not ids: a paste repeats what was copied, even after the
  // originals are edited or deleted.
  const clipboardRef = useRef<DiagramNode[]>([])
  const copySelectedNodes = useCallback(() => {
    const selected = selectedNodes()
    if (selected.length === 0) return false
    clipboardRef.current = selected
    return true
  }, [selectedNodes])
  const pasteNodes = useCallback(() => {
    if (clipboardRef.current.length === 0) return false
    const clones = duplicateNodes(clipboardRef.current, graphRef.current.nodes.map((node) => node.id))
    // The next paste offsets from this one.
    clipboardRef.current = clones
    addClones(clones)
    return true
  }, [addClones])
  useEffect(() => {
    if (readOnly || !active) return
    const onKeyDown = (event: globalThis.KeyboardEvent) => {
      if (!(event.ctrlKey || event.metaKey)) return
      const key = event.key.toLowerCase()
      if (key !== "d" && key !== "c" && key !== "v") return
      const tag = (event.target as HTMLElement | null)?.tagName
      if (tag === "INPUT" || tag === "TEXTAREA") return
      if (key === "d") { event.preventDefault(); duplicateSelectedNodes() }
      else if (key === "c") { if (copySelectedNodes()) event.preventDefault() }
      else if (pasteNodes()) event.preventDefault()
    }
    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [readOnly, active, duplicateSelectedNodes, copySelectedNodes, pasteNodes])
  // Snaps every dragged node, carries nested nodes along (React Flow moves them
  // relative to their parent; the graph stores absolute positions), and a
  // single node dropped into a different compound moves into that element.
  const onNodeDragStop = useCallback((_event: MouseEvent | TouchEvent, _node: GraphFlowNode, dragged: GraphFlowNode[]) => {
    const current = graphRef.current.nodes
    const byId = new Map(current.map((node) => [node.id, node]))
    const shifts = new Map<string, { x: number, y: number }>()
    const placed = new Map<string, { x: number, y: number }>()
    for (const moved of dragged) {
      const node = byId.get(moved.id)
      if (!node) continue
      const origin = node.parent ? byId.get(node.parent)?.position ?? { x: 0, y: 0 } : { x: 0, y: 0 }
      const position = { x: snapToGrid(origin.x + moved.position.x), y: snapToGrid(origin.y + moved.position.y) }
      placed.set(node.id, position)
      shifts.set(node.id, { x: position.x - node.position.x, y: position.y - node.position.y })
    }
    const shiftOf = (node: DiagramNode): { x: number, y: number } | undefined => {
      for (let parent = node.parent; parent; parent = byId.get(parent)?.parent ?? null) {
        const shift = shifts.get(parent)
        if (shift) return shift
      }
      return undefined
    }
    let reparent: { id: string, parent: string | null } | null = null
    const [single] = dragged
    const lone = dragged.length === 1 && single ? byId.get(single.id) : undefined
    if (lone && !lone.compound) {
      const at = placed.get(lone.id)!
      const width = single.measured?.width ?? lone.width ?? FALLBACK_SIZE.width
      const height = single.measured?.height ?? lone.height ?? FALLBACK_SIZE.height
      const parent = innermostNode(current, at.x + width / 2, at.y + height / 2, new Set([lone.id]), true)
      if (parent !== lone.parent) reparent = { id: lone.id, parent }
    }
    emit({
      nodes: current.map((node) => {
        const position = placed.get(node.id)
        const shift = position ? undefined : shiftOf(node)
        const next = position ? { ...node, position } : shift ? { ...node, position: { x: node.position.x + shift.x, y: node.position.y + shift.y } } : node
        return reparent?.id === node.id ? { ...next, parent: reparent.parent } : next
      }),
    })
  }, [emit])
  const onConnect: OnConnect = useCallback((connection: Connection) => {
    if (!connection.source || !connection.target || connection.source === connection.target) return
    // A draft id, so it never matches the `source->target` id of a connection the view already shows.
    const id = nextDraftId(graphRef.current.edges.map((edge) => edge.id))
    emit({ edges: [...graphRef.current.edges, { id, source: connection.source, target: connection.target, relations: [] }] })
  }, [emit])
  // A drop that missed every handle still connects when it lands anywhere on
  // another node, so the target never needs pixel-precise aim.
  const onConnectEnd: OnConnectEnd = useCallback((event, state) => {
    if (state.isValid || !state.fromNode) return
    const point = "changedTouches" in event ? event.changedTouches[0] : event
    if (!point) return
    const nodeEl = document.elementsFromPoint(point.clientX, point.clientY).find((el) => el.classList.contains("react-flow__node"))
    const target = nodeEl?.getAttribute("data-id")
    if (target) onConnect({ source: state.fromNode.id, target, sourceHandle: null, targetHandle: null })
  }, [onConnect])
  // Saved bends mean different things per style (chord offset vs lane offset), so a switch clears them all.
  const setEdgeStyle = useCallback((next: EdgeStyle) => {
    if (next === edgeStyle) return
    onEdgeStyleChange(next)
    emit({ edges: graphRef.current.edges.map(({ path: _path, ...edge }) => edge) })
  }, [edgeStyle, onEdgeStyleChange, emit])
  // Delete never cascades: React Flow would also take a node's connections and
  // children, but only what the user selected leaves the graph. The model then
  // refuses a delete that would leave a reference behind.
  // Nodes follow the view's removal (see `removal`); connections are always model edits.
  const onBeforeDelete: OnBeforeDelete<GraphFlowNode, GraphFlowEdge> = useCallback(async ({ nodes: doomed, edges: doomedEdges }) => {
    const nodeIds = doomed.filter((node) => node.selected).map((node) => node.id)
    const edgeIds = new Set(doomedEdges.filter((edge) => edge.selected).map((edge) => edge.id))
    const edges = edgeIds.size > 0 ? graphRef.current.edges.filter((edge) => !edgeIds.has(edge.id)) : undefined
    if (nodeIds.length > 0 && removal) removal.run(nodeIds, edges)
    else if (edges) emit({ edges })
    return false
  }, [emit, removal])
  // React Flow owns selection.
  const selectedEdgeId = edges.find((edge) => edge.selected)?.id
  const selectedEdge = graph.edges.find((edge) => edge.id === selectedEdgeId)
  const resetSelectedEdgePath = useCallback(() => {
    if (!selectedEdgeId) return
    emit({ edges: graphRef.current.edges.map((edge) => edge.id === selectedEdgeId ? { ...edge, path: undefined } : edge) })
  }, [emit, selectedEdgeId])
  const deleteSelectedEdge = useCallback(() => {
    if (!selectedEdgeId) return
    emit({ edges: graphRef.current.edges.filter((edge) => edge.id !== selectedEdgeId) })
  }, [emit, selectedEdgeId])

  return (
    <div className={cn("architecture-diagram-surface", className)} style={{ display: "flex", minHeight: 0, height: "100%", flexDirection: "column", overflow: "hidden", borderTop: "1px solid var(--divider)", backgroundColor: "var(--sketch-canvas)" }}>
      <div className="architecture-diagram-toolbar">
        {!readOnly && <>
          {topLevelDrawing && <button type="button" onClick={addNode} className="architecture-diagram-toolbar-symbol" aria-label="Add node"><Plus size={13} /></button>}
          <button type="button" onClick={() => setLasso((current) => !current)} aria-pressed={lasso} className={cn("architecture-diagram-toolbar-symbol", lasso && "architecture-diagram-toolbar-symbol-active")} aria-label="Lasso select"><CircleDashed size={13} /></button>
          <span className="architecture-diagram-toolbar-delimiter" aria-hidden="true">|</span>
        </>}
        <button type="button" onClick={fitView} className="architecture-diagram-toolbar-symbol" aria-label="Fit view"><Maximize size={13} /></button>
        {toolbar}
        {!readOnly && <div ref={edgeMenuRef} style={{ position: "relative", marginLeft: "auto" }}>
          <button type="button" onClick={() => setEdgeMenuOpen((current) => !current)} aria-label="Connection style" aria-haspopup="menu" aria-expanded={edgeMenuOpen} className="architecture-diagram-toolbar-symbol"><MoreHorizontal size={15} /></button>
          {edgeMenuOpen && <div role="menu" aria-label="Connection style" style={{ position: "absolute", zIndex: 10, top: "calc(100% + 4px)", right: 0, minWidth: 132, padding: 4, border: "1px solid var(--divider)", borderRadius: 6, background: "var(--paper)", boxShadow: "var(--shadow-lg)" }}>
            <button type="button" role="menuitemradio" aria-checked={edgeStyle === "elbow"} onClick={() => { setEdgeStyle("elbow"); setEdgeMenuOpen(false) }} className="architecture-diagram-menu-item">Angled edges</button>
            <button type="button" role="menuitemradio" aria-checked={edgeStyle === "curved"} onClick={() => { setEdgeStyle("curved"); setEdgeMenuOpen(false) }} className="architecture-diagram-menu-item">Curved edges</button>
          </div>}
        </div>}
      </div>
      <div ref={viewportRef} onWheelCapture={zoomGraphAtCenter} onPointerDown={onPanePointerDown} onContextMenu={(event) => event.preventDefault()} style={{ position: "relative", minHeight: 0, flex: 1 }}>
      {!readOnly && selectedEdge && <div className="architecture-diagram-edge-editor">
        {selectedEdge.path && <button type="button" className="architecture-diagram-icon-button" onClick={resetSelectedEdgePath} aria-label="Reset connection path"><RotateCcw size={14} /></button>}
        <button type="button" className="architecture-diagram-icon-button architecture-diagram-delete" onClick={deleteSelectedEdge} aria-label="Delete connection"><Trash2 size={14} /></button>
      </div>}
      {lasso && !readOnly && <div className="architecture-diagram-draw-overlay" {...lassoOverlay} />}
      <ReactFlow<GraphFlowNode, GraphFlowEdge>
        nodes={flowNodes} edges={edges} nodeTypes={nodeTypes} edgeTypes={edgeTypes}
        onInit={setFlow}
        onNodesChange={handleNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeDragStart={takeCamera} onNodeClick={takeCamera} onPaneClick={takeCamera}
        onMoveStart={(event) => { if (event) takeCamera() }}
        onNodeDragStop={onNodeDragStop} onConnect={onConnect} onConnectEnd={onConnectEnd} connectionRadius={48} onBeforeDelete={onBeforeDelete}
        nodesDraggable={!readOnly} nodesConnectable={!readOnly} elementsSelectable={!readOnly} deleteKeyCode={readOnly ? null : ["Backspace", "Delete"]}
        connectionMode={ConnectionMode.Loose}
        panOnDrag={[0, 1, 2]} zoomOnPinch={readOnly} fitView minZoom={0.2 * hostScale} maxZoom={2 * hostScale} zoomOnScroll={false} zoomOnDoubleClick={false} panOnScroll selectionOnDrag={false} proOptions={{ hideAttribution: true }} elevateEdgesOnSelect
      >
        <Background gap={22} size={1} color="var(--sketch-grid)" />
      </ReactFlow>
      {preview && (
        <svg className="architecture-diagram-draw-preview" aria-hidden="true">
          <path d={preview} fill="none" strokeDasharray={lasso ? "6 4" : undefined} />
        </svg>
      )}
      </div>
    </div>
  )
}
