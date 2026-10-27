import { useCallback, useEffect, useMemo, useState } from "react"
import { edgeAttachments, loadRouter, routeEdges, type EdgeStyle } from "@/lib/domain/edgeRouting"
import type { DiagramEdge } from "./diagramGraph"
import { routeCards, type GraphFlowInstance, type GraphFlowNode } from "./architectureFlow"
import type { EdgeEdits, EdgeGeometry } from "./ArchitectureEdge"

/**
 * Where the surface's connections run: card rectangles from the live flow
 * nodes, edge attachments and libavoid routes, and the edits an edge may make
 * (absent when read-only), including the bend preview of a drag in progress.
 */
export function useEdgeGeometry({ nodes, edges, edgeStyle, flow, readOnly, onPathChange, onLabelChange }: {
  nodes: readonly GraphFlowNode[]
  edges: readonly DiagramEdge[]
  edgeStyle: EdgeStyle
  flow: GraphFlowInstance | null
  readOnly: boolean
  onPathChange: EdgeEdits["onPathChange"]
  onLabelChange: EdgeEdits["onLabelChange"]
}) {
  const toFlowPoint = useCallback((x: number, y: number) => flow?.screenToFlowPosition({ x, y }), [flow])
  const [bendPreview, setBendPreview] = useState<{ id: string, bend: number } | null>(null)
  const previewBend = useCallback((id: string, bend: number | null) => setBendPreview(bend === null ? null : { id, bend }), [])
  const edit = useMemo<EdgeEdits | undefined>(() => readOnly ? undefined : {
    onPathChange,
    onLabelChange,
    toFlowPoint,
    onBendPreview: previewBend,
  }, [onPathChange, onLabelChange, previewBend, readOnly, toFlowPoint])
  // Recomputed from the live node rects, so dragging a card re-slots and reroutes its edges.
  const cards = useMemo(() => routeCards(nodes), [nodes])
  const attachments = useMemo(() => edgeAttachments(cards, edges), [cards, edges])
  // Until libavoid has loaded (or if it cannot), connections take the unrouted shapes.
  const [routerReady, setRouterReady] = useState(false)
  useEffect(() => {
    let live = true
    void loadRouter().then((ready) => { if (live) setRouterReady(ready) })
    return () => { live = false }
  }, [])
  // A ~15-card view routes in about 15 ms.
  const routes = useMemo(() => routerReady
    ? routeEdges(cards, edges.map((edge) => ({
      id: edge.id, source: edge.source, target: edge.target,
      bend: bendPreview?.id === edge.id ? bendPreview.bend : edge.path?.bend ?? 0,
      sourcePoint: attachments.get(edge.id)?.source, targetPoint: attachments.get(edge.id)?.target,
    })))
    : null, [routerReady, cards, edges, attachments, bendPreview])
  const edgeGeometry = useMemo<EdgeGeometry>(() => ({
    edgeStyle, cards: new Map(cards.map((card) => [card.id, card])), attachments, routes, bendPreview, edit,
  }), [edgeStyle, cards, attachments, routes, bendPreview, edit])
  return { cards, edgeGeometry }
}
