"use client"

import { createContext, useContext, useEffect, useMemo, useRef, useState, type KeyboardEvent, type PointerEvent as ReactPointerEvent } from "react"
import { BaseEdge, type EdgeProps } from "@xyflow/react"
import { edgeSketchPaths, roughSeed } from "@/features/editor/architecture/architectureSketch"
import type { DiagramEdge } from "@/features/editor/architecture/diagramGraph"
import { trackPointer } from "@/lib/pointer"
import { useResetOnChange } from "@/hooks/useResetOnChange"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import {
  CORNER_RADIUS, arrowHeadPath, bendPoint, bendThrough, normalOffset, normalPoint, orthogonalRoute, polylineMidpoint, roundedPath,
  type EdgeAttachment, type EdgeStyle, type Point, type RouteCard,
} from "@/lib/domain/edgeRouting"
import { cn } from "@/lib/utils"
import type { GraphFlowEdge } from "./architectureFlow"

/** What the surface lets a user do to a connection; absent when read-only. */
export interface EdgeEdits {
  onPathChange: (id: string, path?: DiagramEdge["path"]) => void
  onLabelChange: (id: string, label: string) => void
  toFlowPoint: (x: number, y: number) => Point | undefined
  /** A bend being dragged, so the route follows the pointer before it is saved; null when the drag ends. */
  onBendPreview: (id: string, bend: number | null) => void
}

/** Everything an edge needs to place itself: the live card rectangles and what was computed from them. */
export interface EdgeGeometry {
  edgeStyle: EdgeStyle
  cards: ReadonlyMap<string, RouteCard>
  attachments: ReadonlyMap<string, EdgeAttachment>
  /** libavoid's routes, clear of every card a connection does not join; null until the router has loaded. */
  routes: ReadonlyMap<string, Point[]> | null
  bendPreview: { id: string, bend: number } | null
  edit?: EdgeEdits
}

export const EdgeGeometryContext = createContext<EdgeGeometry>({ edgeStyle: "elbow", cards: new Map(), attachments: new Map(), routes: null, bendPreview: null })

// A routed connection drawn curved: its corners rounded this far, so it flows instead of stepping.
const CURVE_RADIUS = 48

// A connection label wraps at about this many characters, roughly 90px in the
// 13px hand-drawn font, so it fits the gap between two neighbouring cards.
const LABEL_LINE_CHARS = 12
const LABEL_LINE_HEIGHT = 15

/** A label split into lines of whole words, each at most `LABEL_LINE_CHARS` unless one word is longer. */
function wrapLabel(label: string): string[] {
  const lines: string[] = []
  for (const word of label.split(/\s+/).filter(Boolean)) {
    const last = lines.at(-1)
    if (last !== undefined && last.length + 1 + word.length <= LABEL_LINE_CHARS) lines[lines.length - 1] = `${last} ${word}`
    else lines.push(word)
  }
  return lines
}

/** How dragging or nudging the bend handle moves a connection: along one axis, or off its line in any direction. */
type BendAxis = "x" | "y" | "free"

/** The arrow keys that nudge the bend, and by which sign. */
const NUDGES: Record<BendAxis, Partial<Record<string, 1 | -1>>> = {
  free: { ArrowUp: 1, ArrowDown: -1, ArrowLeft: -1, ArrowRight: 1 },
  x: { ArrowLeft: -1, ArrowRight: 1 },
  y: { ArrowUp: -1, ArrowDown: 1 },
}

interface EdgeShape {
  rough: boolean
  edgePath: string
  arrows: string
  /** Where the bend handle sits; null when the connection cannot be bent. */
  handle: Point | null
  labelPoint: Point
  axis: BendAxis | null
  /** The bend that makes the connection pass this point. */
  project: (point: Point) => number | null
}

// Routed (once libavoid has loaded): libavoid's path, which passes the bend
// point; `bend` offsets that point from the midpoint between the card centres,
// along the normal of the line joining them, as the saved positions store it.
// Unrouted: curved `bend` is relative to the chord (endpoint-to-endpoint line),
// elbow `bend` is the Z's middle-lane offset along its axis. All stay correct as nodes drag.
function edgeShape(elbow: boolean, bend: number, route: readonly Point[] | undefined, attachment: EdgeAttachment | undefined, source: RouteCard | undefined, target: RouteCard | undefined): EdgeShape | null {
  if (route) {
    if (!source || !target) return null
    const through = bend ? bendPoint(source, target, bend) : polylineMidpoint(route)
    return {
      rough: !elbow,
      edgePath: roundedPath(route, elbow ? CORNER_RADIUS : CURVE_RADIUS),
      arrows: arrowHeadPath(route[route.length - 1], route[route.length - 2]),
      handle: through,
      labelPoint: through,
      axis: "free",
      project: (point) => bendThrough(source, target, point),
    }
  }
  if (!attachment) return null
  const { source: from, target: to } = attachment
  if (elbow) {
    const { points, d, axis, handle } = orthogonalRoute(from, to, bend)
    const laneBase = axis ? (from[axis] + to[axis]) / 2 : 0
    return {
      rough: false,
      edgePath: d,
      arrows: arrowHeadPath(points[points.length - 1], points[points.length - 2] ?? from),
      handle,
      labelPoint: handle ?? points[Math.floor(points.length / 2)],
      axis,
      project: (point) => axis ? point[axis] - laneBase : null,
    }
  }
  // A quadratic Bézier does not pass through its control point, so the
  // control sits twice as far out as the point the user actually sees.
  const pathPoint = normalPoint(from, to, bend)
  const control = normalPoint(from, to, 2 * bend)
  return {
    rough: true,
    edgePath: `M ${from.x} ${from.y} Q ${control.x} ${control.y} ${to.x} ${to.y}`,
    arrows: arrowHeadPath(to, control),
    handle: pathPoint,
    labelPoint: pathPoint,
    axis: "free",
    project: (point) => normalOffset(from, to, point),
  }
}

export function ArchitectureEdge({ id, source, target, data, selected }: EdgeProps<GraphFlowEdge>) {
  const { edgeStyle, cards, attachments, routes, bendPreview, edit } = useContext(EdgeGeometryContext)
  const stopDragRef = useRef<(() => void) | null>(null)
  // End a drag still running at unmount so its window listeners do not leak.
  useEffect(() => () => stopDragRef.current?.(), [])
  const [editingLabel, setEditingLabel] = useState(false)
  // Deselecting ends label editing.
  useResetOnChange(selected, (isSelected) => { if (!isSelected) setEditingLabel(false) })
  useEscapeLayer(() => setEditingLabel(false), !!selected && editingLabel && !!edit)
  // The label reaches the model once, when editing ends, not on every keystroke.
  const [labelDraft, setLabelDraft] = useState("")
  const commitLabel = () => {
    setEditingLabel(false)
    if (labelDraft.trim() !== (data?.label ?? "")) edit?.onLabelChange(id, labelDraft.trim())
  }

  const savedBend = data?.path?.bend ?? 0
  const bend = bendPreview?.id === id ? bendPreview.bend : savedBend
  const route = routes?.get(id)
  const attachment = attachments.get(id)
  const sourceCard = cards.get(source)
  const targetCard = cards.get(target)
  const geometry = useMemo(
    () => edgeShape(edgeStyle === "elbow", bend, route, attachment, sourceCard, targetCard),
    [edgeStyle, bend, route, attachment, sourceCard, targetCard],
  )
  const sketch = useMemo(
    () => geometry?.rough ? edgeSketchPaths(geometry.edgePath, roughSeed(id), selected ? 2 : 1.7) : [],
    [geometry, id, selected],
  )

  if (!geometry) return null
  const { labelPoint, handle, axis } = geometry
  const edgeProps = { className: cn("architecture-diagram-edge", selected && "architecture-diagram-edge-selected"), pointerEvents: "none" as const, stroke: "currentColor" }

  const startDrag = (event: ReactPointerEvent<SVGCircleElement>) => {
    event.preventDefault()
    event.stopPropagation()
    // The latest bend, read when the drag ends: a parent update must not run inside a state updater.
    let latest: number | null = null
    stopDragRef.current = trackPointer(
      (pointer) => {
        const point = edit?.toFlowPoint(pointer.clientX, pointer.clientY)
        if (!point) return
        const value = geometry.project(point)
        if (value !== null) {
          latest = value
          edit?.onBendPreview(id, value)
        }
      },
      () => {
        stopDragRef.current = null
        edit?.onBendPreview(id, null)
        if (latest !== null) edit?.onPathChange(id, { bend: latest })
      },
    )
  }
  const nudge = (event: KeyboardEvent<SVGCircleElement>) => {
    const sign = axis && NUDGES[axis][event.key]
    if (!sign) return
    event.preventDefault()
    edit?.onPathChange(id, { bend: savedBend + sign * (event.shiftKey ? 10 : 1) })
  }

  return <>
    <g onDoubleClick={() => { if (edit) { setLabelDraft(data?.label ?? ""); setEditingLabel(true) } }}>
      <BaseEdge id={id} path={geometry.edgePath} interactionWidth={20} style={{ stroke: "transparent" }} />
    </g>
    {geometry.rough
      ? sketch.map((path, index) => <path key={index} d={path.d} fill="none" strokeWidth={path.strokeWidth} strokeLinecap="round" {...edgeProps} />)
      : <path d={geometry.edgePath} fill="none" strokeWidth={selected ? 2 : 1.6} strokeLinecap="round" strokeLinejoin="round" {...edgeProps} />}
    <path d={geometry.arrows} fill="currentColor" strokeWidth={1.2} strokeLinejoin="round" {...edgeProps} />
    {selected && editingLabel && edit
      ? <foreignObject x={labelPoint.x - 70} y={labelPoint.y - 26} width={140} height={24} overflow="visible"><input autoFocus aria-label="Connection label" value={labelDraft} placeholder="Label" onChange={(event) => setLabelDraft(event.target.value)} onBlur={commitLabel} onKeyDown={(event) => { if (event.key === "Enter") { event.preventDefault(); event.currentTarget.blur() } else if (event.key === "Escape") { event.preventDefault(); setEditingLabel(false) } }} className="architecture-diagram-edge-label-input nodrag nopan nowheel" /></foreignObject>
      : data?.label && (
        // Lines stack upwards, so the last sits where a one-line label would.
        <text y={labelPoint.y - 12 - (wrapLabel(data.label).length - 1) * LABEL_LINE_HEIGHT} className="architecture-diagram-edge-label" textAnchor="middle" dominantBaseline="central">
          {wrapLabel(data.label).map((line, index) => <tspan key={index} x={labelPoint.x} dy={index === 0 ? 0 : LABEL_LINE_HEIGHT}>{line}</tspan>)}
        </text>
      )}
    {selected && handle && edit && <circle cx={handle.x} cy={handle.y} r={6} role="button" tabIndex={0} aria-label="Adjust connection curve" aria-keyshortcuts={axis ? Object.keys(NUDGES[axis]).join(" ") : undefined} className="architecture-diagram-path-handle nodrag nopan" onPointerDown={startDrag} onKeyDown={nudge} />}
  </>
}
