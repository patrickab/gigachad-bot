"use client"

import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react"
import {
  Background, ConnectionMode, ReactFlow, useEdgesState, useNodesState,
  type Connection, type OnBeforeDelete, type OnConnect, type OnConnectEnd,
} from "@xyflow/react"
import "@xyflow/react/dist/style.css"
import { CircleDashed, Maximize, MoreHorizontal, Plus, RotateCcw, Trash2 } from "lucide-react"
import { useClickOutside } from "@/hooks/useClickOutside"
import { useSurfaceCamera } from "@/hooks/useSurfaceCamera"
import { useSurfaceDrawing } from "@/hooks/useSurfaceDrawing"
import { isDraft, nextDraftId, type DiagramEdge, type DiagramGraph, type DiagramNode } from "@/lib/diagramGraph"
import { edgeAttachments, loadRouter, routeEdges, type EdgeStyle, type Point, type Rect, type RouteCard } from "@/lib/edgeRouting"
import { cn } from "@/lib/utils"
import { useTabActive } from "@/contexts/TabContext"
import { ArchitectureEdge, EdgeGeometryContext, type EdgeEdits, type EdgeGeometry } from "./ArchitectureEdge"
import { ArchitectureNodeCard } from "./ArchitectureNodeCard"
import { FALLBACK_SIZE, GRID_SIZE, snapToGrid, type ArchitectureNodeData, type GraphFlowEdge, type GraphFlowNode, type NodeRemoval } from "./architectureFlow"

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

/** Stable, so an unset `kinds` prop does not rebuild every node each render. */
const NO_KINDS: readonly string[] = []

// New nodes spawn at a 3:2 width:height ratio.
const NEW_NODE_WIDTH = 240
const NEW_NODE_HEIGHT = 160

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

// Module-level so React Flow does not remount every card on each render.
const nodeTypes = { "architecture-node": ArchitectureNodeCard }
const edgeTypes = { "architecture-edge": ArchitectureEdge }

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
      style: { width: node.width, height: node.height },
      data: { ...node, onChange: onNodeChange, kinds, claimAutoEdit, remove },
    }
  })
}

// Live flow nodes as cards in absolute coordinates, so routing works across nesting levels mid-drag.
// A node React Flow has not measured yet counts at FALLBACK_SIZE.
function routeCards(nodes: readonly GraphFlowNode[]): RouteCard[] {
  const byId = new Map(nodes.map((node) => [node.id, node]))
  const absolute = (node: GraphFlowNode): Point => {
    const parent = node.parentId ? byId.get(node.parentId) : undefined
    if (!parent) return node.position
    const origin = absolute(parent)
    return { x: origin.x + node.position.x, y: origin.y + node.position.y }
  }
  return nodes.map((node) => ({
    id: node.id, ...absolute(node),
    width: node.measured?.width ?? FALLBACK_SIZE.width, height: node.measured?.height ?? FALLBACK_SIZE.height,
    parent: node.parentId ?? null, compound: node.data.compound,
  }))
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
    if (x < node.position.x || y < node.position.y || x > node.position.x + node.width || y > node.position.y + node.height) continue
    if (best === null || node.id.split(".").length > best.split(".").length) best = node.id
  }
  return best
}

function toFlowEdges(edges: DiagramEdge[]): GraphFlowEdge[] {
  return edges.map((edge) => ({ id: edge.id, source: edge.source, target: edge.target, type: "architecture-edge", data: { ...edge } }))
}

// Fields React Flow owns on a node/edge; the graph never describes them.
const RETAINED_BY_FLOW = ["selected", "measured", "dragging"] as const

// Rebuild flow items from the graph, carrying over only what React Flow owns. A
// blind replace clears selection and drops measurements on every edit, making
// edges jump and handles fade mid-interaction. Merging the other way round is
// just as wrong: it would keep keys the rebuilt item deliberately omits, e.g. a
// stale `parentId` after a node is dropped out of its frame.
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

/** Sketch-style editor for a `DiagramGraph`: every edit is emitted through `onChange` as a whole new graph. */
export function ArchitectureDiagramSurface({ graph, onChange, className, readOnly = false, edgeStyle, onEdgeStyleChange, autoFit = false, hostScale = 1, toolbar, onRemoveFromView, select = null, kinds = NO_KINDS, topLevelDrawing = true }: ArchitectureDiagramSurfaceProps) {
  const active = useTabActive()
  // Callbacks read the graph through a ref so they stay stable across renders.
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
  const [flow, setFlow] = useState<Parameters<typeof useSurfaceCamera>[0]["flow"]>(null)
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
  const [bendPreview, setBendPreview] = useState<{ id: string, bend: number } | null>(null)
  const previewBend = useCallback((id: string, bend: number | null) => setBendPreview(bend === null ? null : { id, bend }), [])
  const edit = useMemo<EdgeEdits | undefined>(() => readOnly ? undefined : {
    onPathChange: changeEdgePath,
    onLabelChange: changeEdgeLabel,
    toFlowPoint,
    onBendPreview: previewBend,
  }, [changeEdgePath, changeEdgeLabel, previewBend, readOnly, toFlowPoint])
  const nodeKinds = readOnly ? NO_KINDS : kinds
  const [nodes, setNodes, onNodesChange] = useNodesState<GraphFlowNode>(toFlowNodes(graph.nodes, changeNode, nodeKinds, readOnly ? undefined : claimAutoEdit, removal))
  const [edges, setEdges, onEdgesChange] = useEdgesState<GraphFlowEdge>(toFlowEdges(graph.edges))
  const { fitView, takeCamera, handleNodesChange, zoomGraphAtCenter } = useSurfaceCamera({ flow, viewportRef, onNodesChange, autoFit, hostScale, readOnly })

  // Flow state is a local copy, re-synced from the graph while keeping what React Flow owns.
  useEffect(() => {
    setNodes((current) => reconcile(current, toFlowNodes(graph.nodes, changeNode, nodeKinds, readOnly ? undefined : claimAutoEdit, removal)))
  }, [graph.nodes, changeNode, nodeKinds, claimAutoEdit, readOnly, removal, setNodes])
  useEffect(() => {
    setEdges((current) => reconcile(current, toFlowEdges(graph.edges)))
  }, [graph.edges, setEdges])
  useEffect(() => {
    if (select) setNodes((current) => current.map((node) => ({ ...node, selected: node.id === select })))
  }, [select, setNodes])
  // Only an activated (selected) card moves. Any other card is paper: a drag
  // on it pans (React Flow pans from non-draggable nodes), a pen draws on it,
  // and a click activates it.
  const flowNodes = useMemo(() => nodes.map((node) => node.draggable === !!node.selected ? node : { ...node, draggable: !!node.selected }), [nodes])
  // Recomputed from the live node rects, so dragging a card re-slots and reroutes its edges.
  const cards = useMemo(() => routeCards(nodes), [nodes])
  const attachments = useMemo(() => edgeAttachments(cards, graph.edges), [cards, graph.edges])
  // Until libavoid has loaded (or if it cannot), connections take the unrouted shapes.
  const [routerReady, setRouterReady] = useState(false)
  useEffect(() => {
    let live = true
    void loadRouter().then((ready) => { if (live) setRouterReady(ready) })
    return () => { live = false }
  }, [])
  // A ~15-card view routes in about 15 ms.
  const routes = useMemo(() => routerReady
    ? routeEdges(cards, graph.edges.map((edge) => ({
      id: edge.id, source: edge.source, target: edge.target,
      bend: bendPreview?.id === edge.id ? bendPreview.bend : edge.path?.bend ?? 0,
      sourcePoint: attachments.get(edge.id)?.source, targetPoint: attachments.get(edge.id)?.target,
    })))
    : null, [routerReady, cards, graph.edges, attachments, bendPreview])
  const edgeGeometry = useMemo<EdgeGeometry>(() => ({
    edgeStyle, cards: new Map(cards.map((card) => [card.id, card])), attachments, routes, bendPreview, edit,
  }), [edgeStyle, cards, attachments, routes, bendPreview, edit])

  // A new node lands in the innermost element under its centre, so drawing
  // inside an element creates something inside it. Where nothing frames it, a
  // view without its own canvas element (a saved view) takes nothing.
  const addDraft = useCallback((box: Rect) => {
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
  const { lasso, setLasso, preview, onPanePointerDown, lassoOverlay } = useSurfaceDrawing({ flow, viewportRef, readOnly, cards, setNodes, setEdges, addDraft })
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
    const shifts = new Map<string, Point>()
    const placed = new Map<string, Point>()
    for (const moved of dragged) {
      const node = byId.get(moved.id)
      if (!node) continue
      const origin = node.parent ? byId.get(node.parent)?.position ?? { x: 0, y: 0 } : { x: 0, y: 0 }
      const position = { x: snapToGrid(origin.x + moved.position.x), y: snapToGrid(origin.y + moved.position.y) }
      placed.set(node.id, position)
      shifts.set(node.id, { x: position.x - node.position.x, y: position.y - node.position.y })
    }
    const shiftOf = (node: DiagramNode): Point | undefined => {
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
      const width = single.measured?.width ?? lone.width
      const height = single.measured?.height ?? lone.height
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
  // React Flow would also take a node's connections and children, but only what
  // the user selected is named; the model's delete takes the other uses along.
  // Nodes follow the view's removal (see `removal`); connections are always model edits.
  const onBeforeDelete: OnBeforeDelete<GraphFlowNode, GraphFlowEdge> = useCallback(async ({ nodes: doomed, edges: doomedEdges }) => {
    const nodeIds = doomed.filter((node) => node.selected).map((node) => node.id)
    const edgeIds = new Set(doomedEdges.filter((edge) => edge.selected).map((edge) => edge.id))
    const edges = edgeIds.size > 0 ? graphRef.current.edges.filter((edge) => !edgeIds.has(edge.id)) : undefined
    if (nodeIds.length > 0 && removal) removal.run(nodeIds, edges)
    else if (edges) emit({ edges })
    return false
  }, [emit, removal])
  // Selection lives in React Flow state, not in the graph.
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
      <EdgeGeometryContext.Provider value={edgeGeometry}>
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
          // Editable: wheel and pinch are handled by useSurfaceCamera. Read-only falls back to React Flow's pinch.
          panOnDrag={[0, 1, 2]} zoomOnPinch={readOnly} fitView minZoom={0.2 * hostScale} maxZoom={2 * hostScale} zoomOnScroll={false} zoomOnDoubleClick={false} panOnScroll selectionOnDrag={false} proOptions={{ hideAttribution: true }} elevateEdgesOnSelect
        >
          <Background gap={22} size={1} color="var(--sketch-grid)" />
        </ReactFlow>
      </EdgeGeometryContext.Provider>
      {preview && (
        <svg className="architecture-diagram-draw-preview" aria-hidden="true">
          <path d={preview} fill="none" strokeDasharray={lasso ? "6 4" : undefined} />
        </svg>
      )}
      </div>
    </div>
  )
}
