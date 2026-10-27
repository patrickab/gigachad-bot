"use client"

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react"
import {
  Background, ConnectionMode, ReactFlow, useEdgesState, useNodesState,
  type Connection, type OnBeforeDelete, type OnConnect, type OnConnectEnd, type OnNodeDrag,
} from "@xyflow/react"
import "@xyflow/react/dist/style.css"
import { CircleDashed, Maximize, MoreHorizontal, Plus, RotateCcw, Trash2 } from "lucide-react"
import { useSurfaceCamera } from "@/features/editor/architecture/useSurfaceCamera"
import { useSurfaceDrawing } from "@/features/editor/architecture/useSurfaceDrawing"
import { isDraft, nextDraftId, type DiagramEdge, type DiagramGraph } from "@/features/editor/architecture/diagramGraph"
import type { EdgeStyle, Rect } from "@/lib/domain/edgeRouting"
import { cn } from "@/lib/utils"
import { SEPARATOR, ToolButton } from "@/features/editor/canvas/CanvasToolbar"
import { InTopBar, useHasTopBarSlot } from "@/features/navigation/TopBarSlots"
import { ArchitectureEdge, EdgeGeometryContext } from "./ArchitectureEdge"
import { ArchitectureMenu } from "./ArchitectureMenu"
import { ArchitectureNodeCard } from "./ArchitectureNodeCard"
import { reconcile, toFlowEdges, toFlowNodes, type GraphFlowEdge, type GraphFlowInstance, type GraphFlowNode, type NodePatch, type NodeRemoval } from "./architectureFlow"
import { innermostNode, patchNode, settleDrag } from "./graphEdits"
import { useEdgeGeometry } from "./useEdgeGeometry"
import { useNodeClipboard } from "./useNodeClipboard"

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

// Module-level so React Flow does not remount every card on each render.
const nodeTypes = { "architecture-node": ArchitectureNodeCard }
const edgeTypes = { "architecture-edge": ArchitectureEdge }

/** Sketch-style editor for a `DiagramGraph`: every edit is emitted through `onChange` as a whole new graph. */
export function ArchitectureDiagramSurface({ graph, onChange, className, readOnly = false, edgeStyle, onEdgeStyleChange, autoFit = false, hostScale = 1, toolbar, onRemoveFromView, select = null, kinds = NO_KINDS, topLevelDrawing = true }: ArchitectureDiagramSurfaceProps) {
  // With drawing and view slots the toolbar renders into the top bar, otherwise it sits above the diagram.
  const inTopBar = useHasTopBarSlot("drawing", "view")
  // Callbacks read the graph through a ref so they stay stable across renders.
  const graphRef = useRef(graph)
  useLayoutEffect(() => { graphRef.current = graph }, [graph])
  const emit = useCallback((patch: Partial<DiagramGraph>) => onChange({ ...graphRef.current, ...patch }), [onChange])
  // Cards report positions in React Flow's frame, which is relative to the parent for nested nodes.
  const changeNode = useCallback((id: string, patch: NodePatch) => emit({ nodes: patchNode(graphRef.current.nodes, id, patch) }), [emit])
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
  const changeEdgePath = useCallback((id: string, path?: DiagramEdge["path"]) => emit({ edges: graphRef.current.edges.map((edge) => edge.id === id ? { ...edge, path } : edge) }), [emit])
  const changeEdgeLabel = useCallback((id: string, label: string) => emit({ edges: graphRef.current.edges.map((edge) => edge.id === id ? { ...edge, label } : edge) }), [emit])
  const nodeKinds = readOnly ? NO_KINDS : kinds
  // Filled by the layout effect below before the first paint: the card callbacks read refs, which render must not touch.
  const [nodes, setNodes, onNodesChange] = useNodesState<GraphFlowNode>([])
  const selectNodes = useCallback((ids: ReadonlySet<string>) => setNodes((current) => current.map((node) => ({ ...node, selected: ids.has(node.id) }))), [setNodes])
  const [edges, setEdges, onEdgesChange] = useEdgesState<GraphFlowEdge>(toFlowEdges(graph.edges))
  const { fitView, takeCamera, handleNodesChange, zoomGraphAtCenter } = useSurfaceCamera({ flow, viewportRef, onNodesChange, autoFit, hostScale, readOnly })

  // Flow state is a local copy, re-synced from the graph while keeping what React Flow owns.
  useLayoutEffect(() => {
    setNodes((current) => reconcile(current, toFlowNodes(graph.nodes, changeNode, nodeKinds, readOnly ? undefined : claimAutoEdit, removal)))
  }, [graph.nodes, changeNode, nodeKinds, claimAutoEdit, readOnly, removal, setNodes])
  useEffect(() => {
    setEdges((current) => reconcile(current, toFlowEdges(graph.edges)))
  }, [graph.edges, setEdges])
  useEffect(() => {
    if (select) selectNodes(new Set([select]))
  }, [select, selectNodes])
  // Only an activated (selected) card moves. Any other card is paper: a drag
  // on it pans (React Flow pans from non-draggable nodes), a pen draws on it,
  // and a click activates it.
  const flowNodes = useMemo(() => nodes.map((node) => node.draggable === !!node.selected ? node : { ...node, draggable: !!node.selected }), [nodes])
  const { cards, edgeGeometry } = useEdgeGeometry({ nodes, edges: graph.edges, edgeStyle, flow, readOnly, onPathChange: changeEdgePath, onLabelChange: changeEdgeLabel })

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
  const { lasso, setLasso, preview, onPanePointerDown, lassoOverlay } = useSurfaceDrawing({ flow, viewportRef, readOnly, cards, selectNodes, setEdges, addDraft })
  useNodeClipboard({ graphRef, nodes, emit, selectNodes, enabled: !readOnly })
  const onNodeDragStop: OnNodeDrag<GraphFlowNode> = useCallback((_event, _node, dragged) => emit({ nodes: settleDrag(graphRef.current.nodes, dragged) }), [emit])
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
    emit({ edges: graphRef.current.edges.map((edge) => ({ ...edge, path: undefined })) })
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
  const deleteSelectedEdge = useCallback(() => {
    if (!selectedEdgeId) return
    emit({ edges: graphRef.current.edges.filter((edge) => edge.id !== selectedEdgeId) })
  }, [emit, selectedEdgeId])

  const drawingTools = <>
    {!readOnly && <>
      {topLevelDrawing && <ToolButton icon={Plus} label="Add node" onClick={addNode} />}
      <ToolButton icon={CircleDashed} label="Lasso select" toggle active={lasso} onClick={() => setLasso((current) => !current)} />
      {SEPARATOR}
    </>}
    <ToolButton icon={Maximize} label="Fit view" onClick={fitView} />
    {toolbar}
  </>
  const viewTools = !readOnly && (
    <ArchitectureMenu
      label="Connection style" align="right" menuStyle={{ minWidth: 132 }} className={inTopBar ? undefined : "ml-auto"}
      trigger={(open, toggle) => <ToolButton icon={MoreHorizontal} label="Connection style" active={open} onClick={toggle} aria-haspopup="menu" aria-expanded={open} />}
    >
      {(close) => <>
        <button type="button" role="menuitemradio" aria-checked={edgeStyle === "elbow"} onClick={() => { setEdgeStyle("elbow"); close() }} className="architecture-diagram-menu-item">Angled edges</button>
        <button type="button" role="menuitemradio" aria-checked={edgeStyle === "curved"} onClick={() => { setEdgeStyle("curved"); close() }} className="architecture-diagram-menu-item">Curved edges</button>
      </>}
    </ArchitectureMenu>
  )

  return (
    <div className={cn("architecture-diagram-surface", className)} style={{ display: "flex", minHeight: 0, height: "100%", flexDirection: "column", overflow: "hidden", borderTop: inTopBar ? undefined : "1px solid var(--divider)", backgroundColor: "var(--sketch-canvas)" }}>
      {inTopBar
        ? <><InTopBar name="drawing">{drawingTools}</InTopBar><InTopBar name="view">{viewTools}</InTopBar></>
        : <div className="architecture-diagram-toolbar">{drawingTools}{viewTools}</div>}
      <div ref={viewportRef} onWheelCapture={zoomGraphAtCenter} onPointerDown={onPanePointerDown} onContextMenu={(event) => event.preventDefault()} style={{ position: "relative", minHeight: 0, flex: 1 }}>
      {!readOnly && selectedEdge && <div className="architecture-diagram-edge-editor">
        {selectedEdge.path && <button type="button" className="architecture-diagram-icon-button" onClick={() => changeEdgePath(selectedEdge.id)} aria-label="Reset connection path"><RotateCcw size={14} /></button>}
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
