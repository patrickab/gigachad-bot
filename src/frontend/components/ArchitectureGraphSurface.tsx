"use client"

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type CSSProperties, type KeyboardEvent, type PointerEvent as ReactPointerEvent } from "react"
import {
  Background, BaseEdge, ConnectionMode, Handle, NodeResizer, Position, ReactFlow, useEdgesState, useInternalNode, useNodesState,
  type Connection, type Edge, type EdgeProps, type InternalNode, type Node, type NodeProps, type OnConnect, type ReactFlowInstance,
} from "@xyflow/react"
import "@xyflow/react/dist/style.css"
import rough from "roughjs"
import { Maximize, PenLine, Plus, RotateCcw, Trash2 } from "lucide-react"
import { cn } from "@/lib/utils"
import { useTabActive } from "./TabManager"
import {
  nextArchitectureGraphId,
  type ArchitectureGraph,
  type ArchitectureGraphEdge,
  type ArchitectureGraphEdgePath,
  type ArchitectureGraphNode,
  type ArchitectureGraphNodeShape,
} from "@/lib/architectureGraph"

type GraphFlowNodeData = ArchitectureGraphNode & Record<string, unknown>
interface GraphFlowEdgeData extends ArchitectureGraphEdge, Record<string, unknown> {
  attachment?: EdgeAttachment
  onPathChange?: (id: string, path?: ArchitectureGraphEdgePath) => void
  toFlowPoint?: (x: number, y: number) => { x: number, y: number } | undefined
}
type GraphFlowNode = Node<GraphFlowNodeData, "architecture-node">
type GraphFlowEdge = Edge<GraphFlowEdgeData, "architecture-edge">

// The subset of a React Flow node the attachment pass needs: stored position
// plus whatever React Flow has measured so far.
interface RoutableNode {
  id: string
  position: { x: number, y: number }
  measured?: { width?: number, height?: number }
}

export interface ArchitectureGraphSurfaceProps {
  graph: ArchitectureGraph
  onChange: (graph: ArchitectureGraph) => void
  className?: string
  readOnly?: boolean
  onOpenDocument?: () => void
}

interface ArchitectureNodeData extends ArchitectureGraphNode, Record<string, unknown> {
  onChange: (id: string, patch: Partial<Pick<ArchitectureGraphNode, "title" | "bullets" | "shape" | "width" | "height" | "position">>) => void
}

// Rough.js is the same seeded, multi-stroke renderer Excalidraw uses. Stable
// seeds keep the drawing still across React renders instead of making it jitter.
const roughGenerator = rough.generator()

function roughSeed(id: string): number {
  let hash = 2166136261
  for (let index = 0; index < id.length; index += 1) hash = Math.imul(hash ^ id.charCodeAt(index), 16777619)
  return (hash >>> 0) || 1
}

function roughPaths(d: string, seed: number, strokeWidth = 1.35) {
  return roughGenerator.toPaths(roughGenerator.path(d, {
    seed, roughness: 1.25, bowing: 1, stroke: "currentColor", strokeWidth, preserveVertices: true,
  }))
}

// Mirrors Excalidraw's generic shapes: a rounded rectangle path, plus rough.js's
// own ellipse and polygon generators for circle/diamond form factors.
function nodeSketchPaths(shape: ArchitectureGraphNodeShape, width: number, height: number, seed: number, strokeWidth: number) {
  const options = { seed, roughness: 1.25, bowing: 1, stroke: "currentColor", strokeWidth, preserveVertices: true }
  if (shape === "ellipse") return roughGenerator.toPaths(roughGenerator.ellipse(width / 2, height / 2, width, height, options))
  if (shape === "diamond") {
    const points: [number, number][] = [[width / 2, 0], [width, height / 2], [width / 2, height], [0, height / 2]]
    return roughGenerator.toPaths(roughGenerator.polygon(points, options))
  }
  const radius = Math.max(0, Math.min(9, (width - 2) / 2, (height - 2) / 2))
  const d = `M ${radius} 1 H ${width - radius} Q ${width - 1} 1 ${width - 1} ${radius} V ${height - radius} Q ${width - 1} ${height - 1} ${width - radius} ${height - 1} H ${radius} Q 1 ${height - 1} 1 ${height - radius} V ${radius} Q 1 1 ${radius} 1 Z`
  return roughPaths(d, seed, strokeWidth)
}

interface DrawnShapeResult {
  shape: ArchitectureGraphNodeShape
  box: { x: number, y: number, width: number, height: number }
}

// A small, self-contained recognizer (not ported from Excalidraw's, which has
// its own generic-shape math): fit a drawn stroke's points against a
// rectangle, a diamond, and an ellipse inscribed in its bounding box, and
// pick whichever the points sit closest to. Rectangle is the forgiving
// fallback, since a rough freehand box is the most common intent.
const MIN_DRAW_SIZE = 30
const MAX_OPEN_GAP_RATIO = 0.3
const SHAPE_FIT_TOLERANCE = 0.35

function pointToSegmentDistance(p: { x: number, y: number }, a: { x: number, y: number }, b: { x: number, y: number }): number {
  const length = Math.hypot(b.x - a.x, b.y - a.y) || 1
  return Math.abs((b.x - a.x) * (a.y - p.y) - (a.x - p.x) * (b.y - a.y)) / length
}

export function classifyDrawnShape(points: Array<{ x: number, y: number }>): DrawnShapeResult | null {
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
  if (pathLength === 0 || gap / pathLength > MAX_OPEN_GAP_RATIO) return null // an open stroke is not a closed shape

  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2
  const halfWidth = width / 2 || 1, halfHeight = height / 2 || 1
  const diamondEdges: Array<[{ x: number, y: number }, { x: number, y: number }]> = [
    [{ x: cx, y: minY }, { x: maxX, y: cy }], [{ x: maxX, y: cy }, { x: cx, y: maxY }],
    [{ x: cx, y: maxY }, { x: minX, y: cy }], [{ x: minX, y: cy }, { x: cx, y: minY }],
  ]

  let rectError = 0, diamondError = 0, ellipseError = 0
  for (const p of points) {
    const toRectEdge = Math.min(Math.abs(p.x - minX), Math.abs(p.x - maxX)) / halfWidth
    const toRectEdgeY = Math.min(Math.abs(p.y - minY), Math.abs(p.y - maxY)) / halfHeight
    rectError += Math.min(toRectEdge, toRectEdgeY)
    diamondError += Math.min(...diamondEdges.map(([a, b]) => pointToSegmentDistance(p, a, b))) / Math.min(halfWidth, halfHeight)
    const r = Math.hypot((p.x - cx) / halfWidth, (p.y - cy) / halfHeight)
    ellipseError += Math.abs(r - 1)
  }
  const fits: Array<[ArchitectureGraphNodeShape, number]> = [
    ["rectangle", rectError / points.length],
    ["diamond", diamondError / points.length],
    ["ellipse", ellipseError / points.length],
  ]
  fits.sort((a, b) => a[1] - b[1])
  const [bestShape, bestError] = fits[0]
  const shape = bestError <= SHAPE_FIT_TOLERANCE ? bestShape : "rectangle"
  return { shape, box: { x: minX, y: minY, width, height } }
}

// Structural styling stays inline: React Flow must measure a real box even if
// the stylesheet chunk has not loaded yet. The visible border is an SVG sketch.
const cardStyle: CSSProperties = {
  width: "100%",
  height: "100%",
  position: "relative",
  overflow: "visible",
  border: 0,
  transition: "box-shadow 150ms ease",
}

// Non-rectangle shapes clip their card fill to the sketch outline, drop the
// rectangular shadow, and pad enough that text stays inside the largest
// axis-aligned box the outline can hold (diamonds need much more than ellipses).
const CONTENT_CLIP: Record<ArchitectureGraphNodeShape, CSSProperties> = {
  rectangle: { borderRadius: 8 },
  ellipse: { borderRadius: "50%", padding: "10% 15%" },
  diamond: { clipPath: "polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%)", padding: "26% 28%" },
}
// Rectangle nodes keep a left-aligned header/body split, since a box reads
// naturally as a titled container. Round shapes have no flat top edge for a
// header bar to sit on, so their title and bullets center as one plain block.
const CENTERED_SHAPES = new Set<ArchitectureGraphNodeShape>(["ellipse", "diamond"])

interface PendingFocus {
  index: number
  cursor: number
}

// Quiet time before in-progress node text reaches the graph, so an autosave tick
// persists what is being typed rather than the last blurred value.
const DRAFT_COMMIT_MS = 300
const DEFAULT_NODE_WIDTH = 224
const DEFAULT_NODE_HEIGHT = 96
// Minimum on-screen size a drawn stroke's bounding box must reach to register,
// independent of canvas zoom.
const DRAW_MIN_SCREEN_SIZE = 30
// New nodes spawn at a 3:2 width:height ratio.
const NEW_NODE_WIDTH = 240
const NEW_NODE_HEIGHT = 160
// Drag/resize commits round to this grid, so autosaved positions stay stable
// pixel values instead of accumulating sub-pixel drift across edit sessions.
export const GRID_SIZE = 8
export const snapToGrid = (value: number) => Math.round(value / GRID_SIZE) * GRID_SIZE

// Clones selected nodes with fresh ids, offset diagonally so the copies read
// as new objects rather than sitting exactly on top of the originals.
export function duplicateNodes(existing: ArchitectureGraphNode[], selectedIds: ReadonlySet<string>): ArchitectureGraphNode[] {
  const clones: ArchitectureGraphNode[] = []
  for (const node of existing) {
    if (!selectedIds.has(node.id)) continue
    const id = nextArchitectureGraphId("node", [...existing, ...clones].map((candidate) => candidate.id))
    clones.push({ ...node, id, position: { x: snapToGrid(node.position.x + GRID_SIZE * 3), y: snapToGrid(node.position.y + GRID_SIZE * 3) } })
  }
  return clones
}

function ArchitectureNodeCard({ data, selected }: NodeProps<Node<ArchitectureNodeData, "architecture-node">>) {
  const [hovered, setHovered] = useState(false)
  const [editingTitle, setEditingTitle] = useState(false)
  const titleRef = useRef<HTMLInputElement>(null)
  const cardRef = useRef<HTMLDivElement>(null)
  const [cardSize, setCardSize] = useState({ width: DEFAULT_NODE_WIDTH, height: DEFAULT_NODE_HEIGHT })
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
  const bulletRefs = useRef<Array<HTMLTextAreaElement | null>>([])
  const titleFocusedRef = useRef(false)
  const bulletsFocusedRef = useRef(false)
  const cleanBullet = (line: string) => line.replace(/^\s*[•-]\s?/, "").trim()

  // Typing only touches local drafts; edits commit to the graph on blur, not per keystroke.
  const [titleDraft, setTitleDraft] = useState(data.title)
  useLayoutEffect(() => {
    if (titleFocusedRef.current) return
    setTitleDraft(data.title)
  }, [data.title])
  const [bulletDrafts, setBulletDrafts] = useState<string[]>(() => [...data.bullets])
  useLayoutEffect(() => {
    if (bulletsFocusedRef.current) return
    setBulletDrafts([...data.bullets])
  }, [data.bullets])
  const [pendingFocus, setPendingFocus] = useState<PendingFocus | null>(null)
  useLayoutEffect(() => {
    if (!pendingFocus) return
    const el = bulletRefs.current[pendingFocus.index]
    el?.focus()
    el?.setSelectionRange(pendingFocus.cursor, pendingFocus.cursor)
    setPendingFocus(null)
  }, [pendingFocus])

  const commitTitle = () => data.onChange(data.id, { title: titleDraft })
  const commitBullets = () => data.onChange(data.id, { bullets: bulletDrafts.map(cleanBullet).filter(Boolean) })
  const focusBullet = (index: number, cursor: number) => setPendingFocus({ index, cursor })
  const handleBulletKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>, index: number) => {
    const textarea = event.currentTarget
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault()
      const pos = textarea.selectionStart
      const next = [...bulletDrafts]
      next.splice(index, 1, textarea.value.slice(0, pos), textarea.value.slice(pos))
      setBulletDrafts(next)
      focusBullet(index + 1, 0)
    } else if (event.key === "Backspace" && index > 0 && textarea.selectionStart === 0 && textarea.selectionEnd === 0) {
      event.preventDefault()
      const mergeAt = bulletDrafts[index - 1].length
      const next = [...bulletDrafts]
      next.splice(index - 1, 2, next[index - 1] + next[index])
      setBulletDrafts(next)
      focusBullet(index - 1, mergeAt)
    }
  }
  // Commit drafts while typing too, so autosave never persists a stale node. The
  // focus guards above keep the resulting prop update from moving the caret.
  const commitTitleRef = useRef(commitTitle)
  commitTitleRef.current = commitTitle
  const commitBulletsRef = useRef(commitBullets)
  commitBulletsRef.current = commitBullets
  useEffect(() => {
    if (!titleFocusedRef.current) return
    const id = setTimeout(() => commitTitleRef.current(), DRAFT_COMMIT_MS)
    return () => clearTimeout(id)
  }, [titleDraft])
  useEffect(() => {
    if (!bulletsFocusedRef.current) return
    const id = setTimeout(() => commitBulletsRef.current(), DRAFT_COMMIT_MS)
    return () => clearTimeout(id)
  }, [bulletDrafts])
  const border = "color-mix(in srgb, var(--ink-muted), transparent 68%)"
  const shape: ArchitectureGraphNodeShape = data.shape ?? "rectangle"
  const centered = CENTERED_SHAPES.has(shape)
  const nodeSketch = useMemo(
    () => nodeSketchPaths(shape, cardSize.width, cardSize.height, roughSeed(data.id), selected ? 1.65 : hovered ? 1.5 : 1.3),
    [shape, cardSize, data.id, hovered, selected],
  )
  const controlsVisible = hovered || selected
  // Always interactive so pen/touch (no hover state) can tap a handle; opacity still fades
  // in on mouse hover/selection for a decluttered look, but never blocks the tap target.
  const controlStyle = { opacity: controlsVisible ? 1 : 0.35, pointerEvents: "auto" as const }
  const handleStyle = { ...controlStyle, background: "var(--ink-muted)", borderColor: "var(--surface-elevated)" }
  // Drop refs for removed rows so the autosize pass cannot touch an unmounted textarea.
  bulletRefs.current.length = bulletDrafts.length
  useEffect(() => {
    // scrollHeight forces a sync reflow; defer to next frame to avoid one per keystroke.
    const id = requestAnimationFrame(() => {
      for (const textarea of bulletRefs.current) {
        if (!textarea) continue
        textarea.style.height = "auto"
        textarea.style.height = `${textarea.scrollHeight}px`
      }
    })
    return () => cancelAnimationFrame(id)
  }, [bulletDrafts])
  useLayoutEffect(() => {
    if (editingTitle && titleRef.current && document.activeElement !== titleRef.current) titleRef.current.focus()
  }, [editingTitle])
  return (
    <div ref={cardRef} className={cn("architecture-graph-node", selected && "architecture-graph-node-selected")} style={{ ...cardStyle, backgroundColor: "transparent", boxShadow: shape === "rectangle" ? "var(--architecture-graph-card-shadow)" : "none" }} onMouseEnter={() => setHovered(true)} onMouseLeave={() => setHovered(false)}>
      <NodeResizer isVisible={!!selected} minWidth={80} minHeight={48} color="var(--ink-muted)" onResizeEnd={(_event, params) => {
        const x = snapToGrid(params.x)
        const y = snapToGrid(params.y)
        data.onChange(data.id, { width: snapToGrid(params.x + params.width) - x, height: snapToGrid(params.y + params.height) - y, position: { x, y } })
      }} />
      <svg className="architecture-graph-node-sketch" viewBox={`0 0 ${cardSize.width} ${cardSize.height}`} aria-hidden="true" focusable="false" style={{ position: "absolute", zIndex: 1, inset: 0, width: "100%", height: "100%", overflow: "visible", color: "var(--ink-muted)", pointerEvents: "none" }}>
        {nodeSketch.map((path, index) => <path key={index} d={path.d} fill="none" stroke="currentColor" strokeWidth={path.strokeWidth} />)}
      </svg>
      <Handle id="top" type="target" position={Position.Top} className="architecture-graph-handle" style={handleStyle} />
      <Handle id="left" type="target" position={Position.Left} className="architecture-graph-handle" style={handleStyle} />
      <div style={{ position: "relative", height: "100%", display: "flex", flexDirection: "column", justifyContent: centered ? "center" : undefined, overflow: "hidden", backgroundColor: "var(--surface-elevated)", ...CONTENT_CLIP[shape] }}>
      <div className={cn("architecture-graph-node-titlebar architecture-graph-node-header drag-handle", centered && "architecture-graph-node-titlebar-plain")} style={{ borderBottomColor: centered ? "transparent" : border, backgroundColor: centered ? "transparent" : "var(--surface)" }}>
        {editingTitle ? (
          <input ref={titleRef} autoFocus aria-label="Node title" value={titleDraft} onChange={(event) => setTitleDraft(event.target.value)} onFocus={() => { titleFocusedRef.current = true }} onBlur={() => { titleFocusedRef.current = false; commitTitle(); setEditingTitle(false) }} onKeyDown={(event) => { if (event.key !== "Enter") return; event.preventDefault(); event.currentTarget.blur(); if (bulletDrafts.length === 0) setBulletDrafts([""]); focusBullet(0, 0) }} onPointerDown={(event) => event.stopPropagation()} className="nodrag architecture-graph-title architecture-graph-title-input" style={{ color: "var(--ink-muted)" }} />
        ) : (
          <span role="button" tabIndex={0} aria-label="Edit node title" onClick={() => setEditingTitle(true)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); setEditingTitle(true) } }} className="architecture-graph-title architecture-graph-title-display" style={{ color: "var(--ink-muted)" }}>{data.title || "Untitled node"}</span>
        )}
      </div>
      <div className={cn("architecture-graph-node-body nowheel", centered && "architecture-graph-node-body-plain")}>
        {bulletDrafts.map((text, index) => (
          <div key={index} className="architecture-graph-bullet-row">
            {!centered && <span className="architecture-graph-bullet-marker" aria-hidden="true">—</span>}
            <textarea
              ref={(el) => { bulletRefs.current[index] = el }}
              aria-label="Node bullet"
              value={text}
              onChange={(event) => setBulletDrafts((current) => current.map((line, i) => i === index ? event.target.value : line))}
              onKeyDown={(event) => handleBulletKeyDown(event, index)}
              onFocus={() => { bulletsFocusedRef.current = true }}
              onBlur={() => { bulletsFocusedRef.current = false; commitBullets() }}
              className="nodrag architecture-graph-bullet-input"
              rows={1}
              style={{ color: "var(--ink-muted)", resize: "none" }}
            />
          </div>
        ))}
      </div>
      </div>
      <Handle id="right" type="source" position={Position.Right} className="architecture-graph-handle" style={handleStyle} />
      <Handle id="bottom" type="source" position={Position.Bottom} className="architecture-graph-handle" style={handleStyle} />
    </div>
  )
}

// Floating edges: the attachment point is recomputed from the two nodes' live
// rectangles on every render, so dragging a node re-routes its edges to the
// nearest side instead of leaving a long way-around path behind.

// Pick the side of `node` facing `toward`, returning that side's midpoint. Used
// until the slot pass below has geometry for both cards.
function attach(node: InternalNode<GraphFlowNode>, toward: InternalNode<GraphFlowNode>) {
  const { x, y } = node.internals.positionAbsolute
  const w = node.measured.width ?? DEFAULT_NODE_WIDTH
  const h = node.measured.height ?? DEFAULT_NODE_HEIGHT
  const target = toward.internals.positionAbsolute
  const dx = target.x + (toward.measured.width ?? DEFAULT_NODE_WIDTH) / 2 - (x + w / 2)
  const dy = target.y + (toward.measured.height ?? DEFAULT_NODE_HEIGHT) / 2 - (y + h / 2)
  // Compare against the card's own aspect so wide cards still prefer left/right.
  if (Math.abs(dx) * h > Math.abs(dy) * w) {
    return dx > 0
      ? { x: x + w, y: y + h / 2, position: Position.Right }
      : { x, y: y + h / 2, position: Position.Left }
  }
  return dy > 0
    ? { x: x + w / 2, y: y + h, position: Position.Bottom }
    : { x: x + w / 2, y, position: Position.Top }
}

// Soft avoidance: connections sharing a card side get their own slot on it
// instead of all leaving from the midpoint, so their stepped paths run in
// separate lanes. Lines may still cross — nothing here is a router.
// ponytail: slots are assigned per side, so two edges in the same lane but
// attached to different sides can still overlap. Upgrade path if that shows up
// in practice: a real router (libavoid/ELK) owning the whole diagram.
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

function sideToward(rect: CardRect, toward: CardRect): Position {
  const dx = toward.x + toward.width / 2 - (rect.x + rect.width / 2)
  const dy = toward.y + toward.height / 2 - (rect.y + rect.height / 2)
  // Compare against the card's own aspect so wide cards still prefer left/right.
  if (Math.abs(dx) * rect.height > Math.abs(dy) * rect.width) return dx > 0 ? Position.Right : Position.Left
  return dy > 0 ? Position.Bottom : Position.Top
}

export function edgeAttachments(nodes: readonly RoutableNode[], edges: readonly ArchitectureGraphEdge[]): Map<string, EdgeAttachment> {
  const rects = new Map<string, CardRect>(nodes.map((node) => [node.id, {
    x: node.position.x, y: node.position.y,
    width: node.measured?.width ?? DEFAULT_NODE_WIDTH,
    height: node.measured?.height ?? DEFAULT_NODE_HEIGHT,
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
      const { x, y, width, height } = end.rect
      points.set(`${end.edgeId}:${end.role}`, end.side === Position.Top ? { x: x + width * ratio, y, position: Position.Top }
        : end.side === Position.Bottom ? { x: x + width * ratio, y: y + height, position: Position.Bottom }
        : end.side === Position.Left ? { x, y: y + height * ratio, position: Position.Left }
        : { x: x + width, y: y + height * ratio, position: Position.Right })
    })
  }
  return new Map(edges.flatMap((edge) => {
    const source = points.get(`${edge.id}:source`)
    const target = points.get(`${edge.id}:target`)
    return source && target ? [[edge.id, { source, target }] as const] : []
  }))
}

function arrowHeadPath(tip: { x: number, y: number }, from: { x: number, y: number }): string {
  const angle = Math.atan2(tip.y - from.y, tip.x - from.x)
  const size = 10
  const wing = 0.58
  const left = { x: tip.x - Math.cos(angle - wing) * size, y: tip.y - Math.sin(angle - wing) * size }
  const right = { x: tip.x - Math.cos(angle + wing) * size, y: tip.y - Math.sin(angle + wing) * size }
  return `M ${left.x} ${left.y} L ${tip.x} ${tip.y} L ${right.x} ${right.y}`
}

function ArchitectureEdgePath({ id, source, target, data, selected }: EdgeProps<GraphFlowEdge>) {
  const sourceNode = useInternalNode<GraphFlowNode>(source)
  const targetNode = useInternalNode<GraphFlowNode>(target)
  const [dragBend, setDragBend] = useState<number | null>(null)
  const stopDragRef = useRef<(() => void) | null>(null)
  useEffect(() => () => stopDragRef.current?.(), [])

  // Bend is stored relative to the chord (endpoint-to-endpoint line), not as an
  // absolute point, so it stays correct as the chord moves when nodes drag.
  const geometry = useMemo(() => {
    if (!sourceNode || !targetNode) return null
    const from = data?.attachment?.source ?? attach(sourceNode, targetNode)
    const to = data?.attachment?.target ?? attach(targetNode, sourceNode)
    const mid = { x: (from.x + to.x) / 2, y: (from.y + to.y) / 2 }
    const length = Math.hypot(to.x - from.x, to.y - from.y) || 1
    const normal = { x: -(to.y - from.y) / length, y: (to.x - from.x) / length }
    const bend = dragBend ?? data?.path?.bend ?? 0
    // A quadratic Bézier does not pass through its control point, so the
    // control sits twice as far out as the point the user actually sees.
    const pathPoint = { x: mid.x + normal.x * bend, y: mid.y + normal.y * bend }
    const control = { x: mid.x + normal.x * bend * 2, y: mid.y + normal.y * bend * 2 }
    const edgePath = `M ${from.x} ${from.y} Q ${control.x} ${control.y} ${to.x} ${to.y}`
    const arrows = `${arrowHeadPath(to, control)}${data?.direction === "bidirectional" ? ` ${arrowHeadPath(from, control)}` : ""}`
    return { mid, normal, pathPoint, edgePath, arrows }
  }, [sourceNode, targetNode, data?.attachment, data?.direction, data?.path?.bend, dragBend])

  const sketch = useMemo(
    () => geometry ? roughPaths(`${geometry.edgePath} ${geometry.arrows}`, roughSeed(id), selected ? 1.6 : 1.3) : [],
    [geometry, id, selected],
  )

  if (!geometry) return null
  const { pathPoint } = geometry

  const startDrag = (event: ReactPointerEvent<SVGCircleElement>) => {
    event.preventDefault()
    event.stopPropagation()
    const { mid, normal } = geometry
    const move = (pointer: PointerEvent) => {
      const point = data?.toFlowPoint?.(pointer.clientX, pointer.clientY)
      if (!point) return
      setDragBend((point.x - mid.x) * normal.x + (point.y - mid.y) * normal.y)
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
    const sign = event.key === "ArrowUp" || event.key === "ArrowRight" ? 1
      : event.key === "ArrowDown" || event.key === "ArrowLeft" ? -1 : null
    if (sign === null) return
    event.preventDefault()
    data?.onPathChange?.(id, { bend: (data?.path?.bend ?? 0) + sign * step })
  }

  return <>
    <BaseEdge id={id} path={geometry.edgePath} interactionWidth={20} style={{ stroke: "transparent" }} />
    {sketch.map((path, index) => <path key={index} d={path.d} fill="none" stroke="currentColor" strokeWidth={path.strokeWidth} className={cn("architecture-graph-edge", selected && "architecture-graph-edge-selected")} pointerEvents="none" />)}
    {data?.label && <text x={pathPoint.x} y={pathPoint.y - 12} className="architecture-graph-edge-label" textAnchor="middle" dominantBaseline="central">{data.label}</text>}
    {selected && data?.onPathChange && <circle cx={pathPoint.x} cy={pathPoint.y} r={6} role="button" tabIndex={0} aria-label="Adjust connection curve" aria-keyshortcuts="ArrowUp ArrowDown ArrowLeft ArrowRight" className="architecture-graph-path-handle nodrag nopan" onPointerDown={startDrag} onKeyDown={nudge} />}
  </>
}

const nodeTypes = { "architecture-node": ArchitectureNodeCard }
const edgeTypes = { "architecture-edge": ArchitectureEdgePath }

function toFlowNodes(nodes: ArchitectureGraphNode[], onNodeChange: ArchitectureNodeData["onChange"]): GraphFlowNode[] {
  return nodes.map((node) => ({
    id: node.id, type: "architecture-node", position: node.position, dragHandle: ".architecture-graph-node-header",
    // Height is left unset for legacy nodes without an explicit height, so they
    // keep growing with their content instead of being clamped to a default.
    style: { width: node.width ?? DEFAULT_NODE_WIDTH, ...(node.height !== undefined ? { height: node.height } : {}) },
    data: { ...node, onChange: onNodeChange },
  }))
}

interface EdgeRuntime {
  onPathChange?: GraphFlowEdgeData["onPathChange"]
  toFlowPoint?: GraphFlowEdgeData["toFlowPoint"]
}

function toFlowEdges(edges: ArchitectureGraphEdge[], attachments = new Map<string, EdgeAttachment>(), runtime: EdgeRuntime = {}): GraphFlowEdge[] {
  return edges.map((edge) => ({
    ...edge,
    type: "architecture-edge",
    data: { ...edge, ...runtime, ...(attachments.has(edge.id) ? { attachment: attachments.get(edge.id) } : {}) },
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

export function ArchitectureGraphSurface({ graph, onChange, className, readOnly = false, onOpenDocument }: ArchitectureGraphSurfaceProps) {
  const active = useTabActive()
  const graphRef = useRef(graph)
  graphRef.current = graph
  const emit = useCallback((patch: Partial<ArchitectureGraph>) => onChange({ ...graphRef.current, ...patch }), [onChange])
  const changeNode = useCallback((id: string, patch: Partial<Pick<ArchitectureGraphNode, "title" | "bullets" | "shape" | "width" | "height" | "position">>) => {
    emit({ nodes: graphRef.current.nodes.map((node) => node.id === id ? { ...node, ...patch } : node) })
  }, [emit])
  const [flow, setFlow] = useState<ReactFlowInstance<GraphFlowNode, GraphFlowEdge> | null>(null)
  const changeEdgePath = useCallback((id: string, path?: ArchitectureGraphEdgePath) => {
    emit({ edges: graphRef.current.edges.map((edge) => edge.id === id ? { ...edge, path } : edge) })
  }, [emit])
  const toFlowPoint = useCallback((x: number, y: number) => flow?.screenToFlowPosition({ x, y }), [flow])
  const edgeRuntime = useMemo<EdgeRuntime>(() => readOnly ? {} : {
    onPathChange: changeEdgePath,
    toFlowPoint,
  }, [changeEdgePath, readOnly, toFlowPoint])
  const [nodes, setNodes, onNodesChange] = useNodesState<GraphFlowNode>(toFlowNodes(graph.nodes, changeNode))
  const [edges, setEdges, onEdgesChange] = useEdgesState<GraphFlowEdge>(toFlowEdges(graph.edges, new Map(), edgeRuntime))

  useEffect(() => {
    setNodes((current) => reconcile(current, toFlowNodes(graph.nodes, changeNode)))
  }, [graph.nodes, changeNode, setNodes])
  // Recomputed from the live node rects, so dragging a card re-slots its edges.
  const attachments = useMemo(() => edgeAttachments(nodes, graph.edges), [nodes, graph.edges])
  useEffect(() => {
    setEdges((current) => reconcile(current, toFlowEdges(graph.edges, attachments, edgeRuntime)))
  }, [graph.edges, attachments, edgeRuntime, setEdges])

  const addNode = useCallback(() => {
    const id = nextArchitectureGraphId("node", graphRef.current.nodes.map((node) => node.id))
    const node: ArchitectureGraphNode = { id, title: "New node", bullets: [], position: { x: 100 + graphRef.current.nodes.length * 28, y: 100 + graphRef.current.nodes.length * 28 }, width: NEW_NODE_WIDTH, height: NEW_NODE_HEIGHT }
    emit({ nodes: [...graphRef.current.nodes, node] })
  }, [emit])
  const commitDrawnShape = useCallback((shape: ArchitectureGraphNodeShape, box: { x: number, y: number, width: number, height: number }) => {
    const id = nextArchitectureGraphId("node", graphRef.current.nodes.map((node) => node.id))
    const node: ArchitectureGraphNode = { id, title: "New node", bullets: [], position: { x: box.x, y: box.y }, shape, width: Math.max(80, Math.round(box.width)), height: Math.max(48, Math.round(box.height)) }
    emit({ nodes: [...graphRef.current.nodes, node] })
  }, [emit])
  const [drawMode, setDrawMode] = useState(false)
  useEffect(() => {
    if (!drawMode) return
    const onKeyDown = (event: globalThis.KeyboardEvent) => { if (event.key === "Escape") setDrawMode(false) }
    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [drawMode])
  // Points accumulate in a ref so every pointer sample doesn't re-render the
  // whole surface; the preview only syncs to state once per animation frame.
  const drawPointsRef = useRef<Array<{ x: number, y: number }>>([])
  const drawFrameRef = useRef<number | null>(null)
  const [drawPreview, setDrawPreview] = useState<Array<{ x: number, y: number }> | null>(null)
  const drawOriginRef = useRef({ left: 0, top: 0 })
  useEffect(() => () => { if (drawFrameRef.current !== null) cancelAnimationFrame(drawFrameRef.current) }, [])
  const onDrawPointerDown = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId)
    const rect = event.currentTarget.getBoundingClientRect()
    drawOriginRef.current = { left: rect.left, top: rect.top }
    drawPointsRef.current = [{ x: event.clientX, y: event.clientY }]
    setDrawPreview(drawPointsRef.current)
  }, [])
  const onDrawPointerMove = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    if (drawPointsRef.current.length === 0) return
    drawPointsRef.current.push({ x: event.clientX, y: event.clientY })
    if (drawFrameRef.current !== null) return
    drawFrameRef.current = requestAnimationFrame(() => {
      drawFrameRef.current = null
      setDrawPreview([...drawPointsRef.current])
    })
  }, [])
  // Zoom-independent: the size gate runs on screen pixels, before the points
  // are converted to flow space (where a low zoom would shrink the gesture
  // needed to register and a high zoom would inflate it).
  const finishDraw = useCallback((commit: boolean) => {
    if (drawFrameRef.current !== null) { cancelAnimationFrame(drawFrameRef.current); drawFrameRef.current = null }
    const points = drawPointsRef.current
    drawPointsRef.current = []
    setDrawPreview(null)
    if (!commit || !flow || points.length === 0) return
    const xs = points.map((p) => p.x)
    const ys = points.map((p) => p.y)
    if (Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) < DRAW_MIN_SCREEN_SIZE) return
    const result = classifyDrawnShape(points.map((p) => flow.screenToFlowPosition(p)))
    if (result) { commitDrawnShape(result.shape, result.box); setDrawMode(false) }
  }, [flow, commitDrawnShape])
  const onDrawPointerUp = useCallback(() => finishDraw(true), [finishDraw])
  const onDrawPointerCancel = useCallback(() => finishDraw(false), [finishDraw])
  const duplicateSelectedNodes = useCallback(() => {
    const selectedIds = new Set(nodes.filter((node) => node.selected).map((node) => node.id))
    if (selectedIds.size === 0) return
    const clones = duplicateNodes(graphRef.current.nodes, selectedIds)
    if (clones.length === 0) return
    const cloneIds = new Set(clones.map((clone) => clone.id))
    emit({ nodes: [...graphRef.current.nodes, ...clones] })
    // Select the copies, not the originals, so repeated Ctrl+D walks diagonally
    // instead of stacking every clone on the same spot.
    setNodes((current) => current.map((node) => ({ ...node, selected: cloneIds.has(node.id) })))
  }, [nodes, emit, setNodes])
  useEffect(() => {
    if (readOnly || !active) return
    const onKeyDown = (event: globalThis.KeyboardEvent) => {
      if (!(event.ctrlKey || event.metaKey) || event.key.toLowerCase() !== "d") return
      const tag = (event.target as HTMLElement | null)?.tagName
      if (tag === "INPUT" || tag === "TEXTAREA") return
      event.preventDefault()
      duplicateSelectedNodes()
    }
    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [readOnly, active, duplicateSelectedNodes])
  const onNodeDragStop = useCallback((_event: MouseEvent | TouchEvent, moved: GraphFlowNode) => {
    const position = { x: snapToGrid(moved.position.x), y: snapToGrid(moved.position.y) }
    emit({ nodes: graphRef.current.nodes.map((node) => node.id === moved.id ? { ...node, position } : node) })
  }, [emit])
  const onConnect: OnConnect = useCallback((connection: Connection) => {
    if (!connection.source || !connection.target || connection.source === connection.target) return
    const id = nextArchitectureGraphId("edge", graphRef.current.edges.map((edge) => edge.id))
    const edge: ArchitectureGraphEdge = { id, source: connection.source, target: connection.target, direction: "one-way" }
    emit({ edges: [...graphRef.current.edges, edge] })
  }, [emit])
  const onEdgesDelete = useCallback((deleted: Edge[]) => emit({ edges: graphRef.current.edges.filter((edge) => !deleted.some((item) => item.id === edge.id)) }), [emit])
  const onNodesDelete = useCallback((deleted: Node[]) => {
    const ids = new Set(deleted.map((node) => node.id))
    emit({ nodes: graphRef.current.nodes.filter((node) => !ids.has(node.id)), edges: graphRef.current.edges.filter((edge) => !ids.has(edge.source) && !ids.has(edge.target)) })
  }, [emit])
  // React Flow owns selection; the inspector just follows it.
  const selectedEdgeId = edges.find((edge) => edge.selected)?.id
  const selectedEdge = graph.edges.find((edge) => edge.id === selectedEdgeId)
  const updateSelectedEdge = useCallback((patch: Partial<Pick<ArchitectureGraphEdge, "label" | "direction" | "path">>) => {
    if (!selectedEdgeId) return
    emit({ edges: graphRef.current.edges.map((edge) => edge.id === selectedEdgeId ? { ...edge, ...patch } : edge) })
  }, [emit, selectedEdgeId])
  const deleteSelectedEdge = useCallback(() => {
    if (!selectedEdgeId) return
    emit({ edges: graphRef.current.edges.filter((edge) => edge.id !== selectedEdgeId) })
  }, [emit, selectedEdgeId])
  const fitGraph = useCallback(() => flow?.fitView({ padding: 0.22, duration: 180 }), [flow])

  return (
    <div className={cn("architecture-graph-surface", className)} style={{ display: "flex", minHeight: 280, height: "100%", flexDirection: "column", overflow: "hidden", borderTop: "1px solid var(--divider)", backgroundColor: "var(--architecture-graph-canvas)" }}>
      <div className="architecture-graph-toolbar" style={{ position: "relative", inset: "auto", zIndex: 5, display: "flex", minHeight: 31, flexShrink: 0, alignItems: "center", gap: 6, borderBottom: "1px solid var(--divider)", padding: "0 10px" }}>
        {!readOnly && <>
          <button type="button" onClick={addNode} className="architecture-graph-toolbar-symbol" aria-label="Add node"><Plus size={13} /></button>
          <button type="button" onClick={() => setDrawMode((current) => !current)} aria-pressed={drawMode} className={cn("architecture-graph-toolbar-symbol", drawMode && "architecture-graph-toolbar-symbol-active")} aria-label="Draw a shape"><PenLine size={13} /></button>
          <span className="architecture-graph-toolbar-delimiter" aria-hidden="true">|</span>
        </>}
        <button type="button" onClick={fitGraph} className="architecture-graph-toolbar-symbol" aria-label="Fit view"><Maximize size={13} /></button>
        {onOpenDocument && <button type="button" onClick={onOpenDocument} className="architecture-graph-icon-button" aria-label="Open Architecture Graph document" style={{ display: "grid", width: 26, height: 26, marginLeft: "auto", placeItems: "center", border: 0, borderRadius: 5, background: "transparent", color: "var(--ink-muted)" }}><Maximize size={14} /></button>}
      </div>
      <div style={{ position: "relative", minHeight: 0, flex: 1 }}>
      {!readOnly && selectedEdge && <div className="architecture-graph-edge-editor">
        <input aria-label="Connection label" value={selectedEdge.label ?? ""} placeholder="Connection label" onChange={(event) => updateSelectedEdge({ label: event.target.value })} />
        <select aria-label="Connection direction" value={selectedEdge.direction} onChange={(event) => updateSelectedEdge({ direction: event.target.value as ArchitectureGraphEdge["direction"] })}>
          <option value="one-way">One-way</option>
          <option value="bidirectional">Bidirectional</option>
        </select>
        {selectedEdge.path && <button type="button" className="architecture-graph-icon-button" onClick={() => updateSelectedEdge({ path: undefined })} aria-label="Reset connection path"><RotateCcw size={14} /></button>}
        <button type="button" className="architecture-graph-icon-button architecture-graph-delete" onClick={deleteSelectedEdge} aria-label="Delete connection"><Trash2 size={14} /></button>
      </div>}
      {drawMode && !readOnly && (
        <div
          className="architecture-graph-draw-overlay"
          onPointerDown={onDrawPointerDown}
          onPointerMove={onDrawPointerMove}
          onPointerUp={onDrawPointerUp}
          onPointerCancel={onDrawPointerCancel}
        >
          {drawPreview && drawPreview.length > 1 && (
            <svg className="architecture-graph-draw-preview" aria-hidden="true">
              <path
                d={`M ${drawPreview.map((p) => `${p.x - drawOriginRef.current.left} ${p.y - drawOriginRef.current.top}`).join(" L ")}`}
                fill="none"
              />
            </svg>
          )}
        </div>
      )}
      <ReactFlow<GraphFlowNode, GraphFlowEdge>
        nodes={nodes} edges={edges} nodeTypes={nodeTypes} edgeTypes={edgeTypes}
        onInit={setFlow}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeDragStop={onNodeDragStop} onConnect={onConnect} onEdgesDelete={onEdgesDelete} onNodesDelete={onNodesDelete}
        nodesDraggable={!readOnly} nodesConnectable={!readOnly} elementsSelectable={!readOnly} deleteKeyCode={readOnly ? null : ["Backspace", "Delete"]}
        connectionMode={ConnectionMode.Loose}
        fitView minZoom={0.2} maxZoom={2} panOnScroll selectionOnDrag={false} proOptions={{ hideAttribution: true }} elevateEdgesOnSelect
      >
        <Background gap={22} size={1} color="var(--divider)" />
      </ReactFlow>
      </div>
    </div>
  )
}
