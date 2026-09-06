"use client"

import { useCallback, useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type KeyboardEvent } from "react"
import {
  Background, BaseEdge, ConnectionMode, getSmoothStepPath, Handle, MarkerType, Position, ReactFlow, useEdgesState, useInternalNode, useNodesState,
  type Connection, type Edge, type EdgeProps, type InternalNode, type Node, type NodeProps, type OnConnect, type ReactFlowInstance,
} from "@xyflow/react"
import "@xyflow/react/dist/style.css"
import { Maximize, Plus, Trash2 } from "lucide-react"
import { cn } from "@/lib/utils"
import {
  nextArchitectureGraphId,
  type ArchitectureGraph,
  type ArchitectureGraphEdge,
  type ArchitectureGraphNode,
} from "@/lib/architectureGraph"

type GraphFlowNodeData = ArchitectureGraphNode & Record<string, unknown>
type GraphFlowEdgeData = ArchitectureGraphEdge & Record<string, unknown>
type GraphFlowNode = Node<GraphFlowNodeData, "architecture-node">
type GraphFlowEdge = Edge<GraphFlowEdgeData, "architecture-edge">

export interface ArchitectureGraphSurfaceProps {
  graph: ArchitectureGraph
  onChange: (graph: ArchitectureGraph) => void
  className?: string
  readOnly?: boolean
  onOpenDocument?: () => void
}

interface ArchitectureNodeData extends ArchitectureGraphNode, Record<string, unknown> {
  onChange: (id: string, patch: Partial<Pick<ArchitectureGraphNode, "title" | "bullets">>) => void
}

// Structural styling stays inline: it must not depend on the stylesheet chunk
// rebuilding in lockstep, or React Flow measures a zero-height box and hides
// every node. Colour comes from role tokens, which cascade on their own.
const cardStyle: CSSProperties = {
  width: "100%",
  overflow: "hidden",
  border: "1px solid var(--architecture-graph-card-border)",
  borderRadius: 8,
  boxShadow: "var(--architecture-graph-card-shadow)",
  transition: "border-color 150ms ease, box-shadow 150ms ease",
}

interface PendingFocus {
  index: number
  cursor: number
}

// Quiet time before in-progress node text reaches the graph, so an autosave tick
// persists what is being typed rather than the last blurred value.
const DRAFT_COMMIT_MS = 300

function ArchitectureNodeCard({ data, selected }: NodeProps<Node<ArchitectureNodeData, "architecture-node">>) {
  const [hovered, setHovered] = useState(false)
  const [editingTitle, setEditingTitle] = useState(false)
  const titleRef = useRef<HTMLInputElement>(null)
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
  const hoverBorder = "color-mix(in srgb, var(--ink-muted), transparent 54%)"
  const selectedBorder = "color-mix(in srgb, var(--ink-muted), transparent 42%)"
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
    <div className={cn("architecture-graph-node", selected && "architecture-graph-node-selected")} style={{ ...cardStyle, backgroundColor: "var(--surface-elevated)", borderColor: selected ? selectedBorder : hovered ? hoverBorder : border }} onMouseEnter={() => setHovered(true)} onMouseLeave={() => setHovered(false)}>
      <Handle id="top" type="target" position={Position.Top} className="architecture-graph-handle" style={handleStyle} />
      <Handle id="left" type="target" position={Position.Left} className="architecture-graph-handle" style={handleStyle} />
      <div className="architecture-graph-node-titlebar architecture-graph-node-header drag-handle" style={{ borderBottomColor: border, backgroundColor: "var(--surface)" }}>
        {editingTitle ? (
          <input ref={titleRef} autoFocus aria-label="Node title" value={titleDraft} onChange={(event) => setTitleDraft(event.target.value)} onFocus={() => { titleFocusedRef.current = true }} onBlur={() => { titleFocusedRef.current = false; commitTitle(); setEditingTitle(false) }} onKeyDown={(event) => { if (event.key !== "Enter") return; event.preventDefault(); event.currentTarget.blur(); if (bulletDrafts.length === 0) setBulletDrafts([""]); focusBullet(0, 0) }} onPointerDown={(event) => event.stopPropagation()} className="nodrag architecture-graph-title architecture-graph-title-input" style={{ color: "var(--ink-muted)" }} />
        ) : (
          <span role="button" tabIndex={0} aria-label="Edit node title" onClick={() => setEditingTitle(true)} onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); setEditingTitle(true) } }} className="architecture-graph-title architecture-graph-title-display" style={{ color: "var(--ink-muted)" }}>{data.title || "Untitled node"}</span>
        )}
      </div>
      <div className="architecture-graph-node-body">
        {bulletDrafts.map((text, index) => (
          <div key={index} className="architecture-graph-bullet-row">
            <span className="architecture-graph-bullet-marker" aria-hidden="true">—</span>
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
      <Handle id="right" type="source" position={Position.Right} className="architecture-graph-handle" style={handleStyle} />
      <Handle id="bottom" type="source" position={Position.Bottom} className="architecture-graph-handle" style={handleStyle} />
    </div>
  )
}

// Floating edges: the attachment point is recomputed from the two nodes' live
// rectangles on every render, so dragging a node re-routes its edges to the
// nearest side instead of leaving a long way-around path behind.
const DEFAULT_NODE_WIDTH = 224
const DEFAULT_NODE_HEIGHT = 96

// Pick the side of `node` facing `toward`, returning that side's midpoint —
// where orthogonal (smoothstep) routing wants to start and end.
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

function ArchitectureEdgePath({ id, source, target, data, selected, markerEnd, markerStart }: EdgeProps<GraphFlowEdge>) {
  const sourceNode = useInternalNode<GraphFlowNode>(source)
  const targetNode = useInternalNode<GraphFlowNode>(target)
  if (!sourceNode || !targetNode) return null

  const from = attach(sourceNode, targetNode)
  const to = attach(targetNode, sourceNode)
  const [edgePath, labelX, labelY] = getSmoothStepPath({
    sourceX: from.x, sourceY: from.y, sourcePosition: from.position,
    targetX: to.x, targetY: to.y, targetPosition: to.position,
  })
  return <>
    <BaseEdge id={id} path={edgePath} markerEnd={markerEnd} markerStart={markerStart} className={cn("architecture-graph-edge", selected && "architecture-graph-edge-selected")} />
    {data?.label && <text x={labelX} y={labelY} className="architecture-graph-edge-label" textAnchor="middle" dominantBaseline="central">{data.label}</text>}
  </>
}

const nodeTypes = { "architecture-node": ArchitectureNodeCard }
const edgeTypes = { "architecture-edge": ArchitectureEdgePath }

function toFlowNodes(nodes: ArchitectureGraphNode[], onNodeChange: ArchitectureNodeData["onChange"]): GraphFlowNode[] {
  return nodes.map((node) => ({
    id: node.id, type: "architecture-node", position: node.position, dragHandle: ".architecture-graph-node-header",
    style: { width: 224 },
    data: { ...node, onChange: onNodeChange },
  }))
}

function toFlowEdges(edges: ArchitectureGraphEdge[]): GraphFlowEdge[] {
  return edges.map((edge) => ({ ...edge, type: "architecture-edge", data: edge as GraphFlowEdgeData, markerEnd: { type: MarkerType.ArrowClosed, color: "var(--ink-muted)" }, ...(edge.direction === "bidirectional" ? { markerStart: { type: MarkerType.ArrowClosed, color: "var(--ink-muted)" } } : {}) }))
}

// Fields React Flow owns on a node/edge; the graph never describes them.
const RETAINED_BY_FLOW = ["selected", "measured", "dragging"] as const

// Rebuild flow items from the graph, carrying over only what React Flow owns. A
// blind replace clears selection and drops measurements on every edit, making
// edges jump and handles fade mid-interaction. Merging the other way round is
// just as wrong: it would keep keys the rebuilt item deliberately omits, e.g. a
// stale markerStart after a bidirectional edge becomes one-way.
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
  const graphRef = useRef(graph)
  graphRef.current = graph
  const emit = useCallback((patch: Partial<ArchitectureGraph>) => onChange({ ...graphRef.current, ...patch }), [onChange])
  const changeNode = useCallback((id: string, patch: Partial<Pick<ArchitectureGraphNode, "title" | "bullets">>) => {
    emit({ nodes: graphRef.current.nodes.map((node) => node.id === id ? { ...node, ...patch } : node) })
  }, [emit])
  const [nodes, setNodes, onNodesChange] = useNodesState<GraphFlowNode>(toFlowNodes(graph.nodes, changeNode))
  const [edges, setEdges, onEdgesChange] = useEdgesState<GraphFlowEdge>(toFlowEdges(graph.edges))
  const [flow, setFlow] = useState<ReactFlowInstance<GraphFlowNode, GraphFlowEdge> | null>(null)

  useEffect(() => {
    setNodes((current) => reconcile(current, toFlowNodes(graph.nodes, changeNode)))
  }, [graph.nodes, changeNode, setNodes])
  useEffect(() => {
    setEdges((current) => reconcile(current, toFlowEdges(graph.edges)))
  }, [graph.edges, setEdges])

  const addNode = useCallback(() => {
    const id = nextArchitectureGraphId("node", graphRef.current.nodes.map((node) => node.id))
    const node: ArchitectureGraphNode = { id, title: "New node", bullets: [], position: { x: 100 + graphRef.current.nodes.length * 28, y: 100 + graphRef.current.nodes.length * 28 } }
    emit({ nodes: [...graphRef.current.nodes, node] })
  }, [emit])
  const onNodeDragStop = useCallback((_event: MouseEvent | TouchEvent, moved: GraphFlowNode) => {
    emit({ nodes: graphRef.current.nodes.map((node) => node.id === moved.id ? { ...node, position: moved.position } : node) })
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
  const updateSelectedEdge = useCallback((patch: Partial<Pick<ArchitectureGraphEdge, "label" | "direction">>) => {
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
          <button type="button" onClick={addNode} className="architecture-graph-toolbar-symbol" aria-label="Add node" title="Add node"><Plus size={13} /></button>
          <span className="architecture-graph-toolbar-delimiter" aria-hidden="true">|</span>
        </>}
        <button type="button" onClick={fitGraph} className="architecture-graph-toolbar-symbol" aria-label="Fit view" title="Fit view"><Maximize size={13} /></button>
        {onOpenDocument && <button type="button" onClick={onOpenDocument} className="architecture-graph-icon-button" aria-label="Open Architecture Graph document" style={{ display: "grid", width: 26, height: 26, marginLeft: "auto", placeItems: "center", border: 0, borderRadius: 5, background: "transparent", color: "var(--ink-muted)" }}><Maximize size={14} /></button>}
      </div>
      <div style={{ position: "relative", minHeight: 0, flex: 1 }}>
      {!readOnly && selectedEdge && <div className="architecture-graph-edge-editor">
        <input aria-label="Connection label" value={selectedEdge.label ?? ""} placeholder="Connection label" onChange={(event) => updateSelectedEdge({ label: event.target.value })} />
        <select aria-label="Connection direction" value={selectedEdge.direction} onChange={(event) => updateSelectedEdge({ direction: event.target.value as ArchitectureGraphEdge["direction"] })}>
          <option value="one-way">One-way</option>
          <option value="bidirectional">Bidirectional</option>
        </select>
        <button type="button" className="architecture-graph-icon-button architecture-graph-delete" onClick={deleteSelectedEdge} aria-label="Delete connection"><Trash2 size={14} /></button>
      </div>}
      <ReactFlow<GraphFlowNode, GraphFlowEdge>
        nodes={nodes} edges={edges} nodeTypes={nodeTypes} edgeTypes={edgeTypes}
        onInit={setFlow}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeDragStop={onNodeDragStop} onConnect={onConnect} onEdgesDelete={onEdgesDelete} onNodesDelete={onNodesDelete}
        nodesDraggable={!readOnly} nodesConnectable={!readOnly} elementsSelectable={!readOnly} deleteKeyCode={readOnly ? null : ["Backspace", "Delete"]}
        connectionMode={ConnectionMode.Loose}
        fitView minZoom={0.2} maxZoom={2} panOnScroll selectionOnDrag={false} proOptions={{ hideAttribution: true }}
      >
        <Background gap={22} size={1} color="var(--divider)" />
      </ReactFlow>
      </div>
    </div>
  )
}
