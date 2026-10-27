"use client"

import { useMemo, useRef, useState, type CSSProperties } from "react"
import { Handle, NodeResizer, Position, type NodeProps } from "@xyflow/react"
import { X } from "lucide-react"
import { isDraft } from "@/features/editor/architecture/diagramGraph"
import { nodeSketchPaths, roughSeed } from "@/features/editor/architecture/architectureSketch"
import { cn } from "@/lib/utils"
import { useElementSize } from "@/hooks/useElementSize"
import { StyledSelect } from "@/components/ui/StyledSelect"
import { FALLBACK_SIZE, snapToGrid, type GraphFlowNode } from "./architectureFlow"
import { useCardFit } from "./useCardFit"
import { useCardText } from "./useCardText"

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

export function ArchitectureNodeCard({ data, selected }: NodeProps<GraphFlowNode>) {
  const [hovered, setHovered] = useState(false)
  const cardRef = useRef<HTMLDivElement>(null)
  // Track the rendered size so the sketch outline always matches the card.
  const cardSize = useElementSize(cardRef, FALLBACK_SIZE)
  const { titleRef, bulletRefs, editingTitle, setEditingTitle, titleDraft, setTitleDraft, bulletDrafts, editBullet, startBullet, handleCardBlur, handleTitleKeyDown, handleBulletKeyDown } = useCardText(data, cardRef)
  const { titlebarRef, bodyRef, resizingRef, fitHeight, minNodeWidth } = useCardFit(data, { titleDraft, bulletDrafts, editingTitle, bulletRefs, cardSize })
  const nodeSketch = useMemo(() => {
    const paths = nodeSketchPaths(Math.max(1, cardSize.width), Math.max(1, cardSize.height), roughSeed(data.id), selected ? 1.9 : hovered ? 1.7 : 1.5)
    return { fills: paths.filter((path) => path.fill !== "none"), strokes: paths.filter((path) => path.fill === "none") }
  }, [cardSize, data.id, hovered, selected])
  const controlsVisible = hovered || selected
  // Hidden at rest so a sketch stays a sketch. Still always interactive, so
  // pen/touch (no hover state) can tap a handle; a tap that selects the node
  // reveals them.
  const controlStyle = { opacity: controlsVisible ? 1 : 0, pointerEvents: "auto" as const }
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
