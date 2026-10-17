"use client"

import { useEffect, useEffectEvent, useLayoutEffect, useMemo, useRef, useState, type CSSProperties, type FocusEvent, type KeyboardEvent, type RefObject } from "react"
import { Handle, NodeResizer, Position, type NodeProps } from "@xyflow/react"
import { X } from "lucide-react"
import { isDraft } from "@/features/editor/architecture/diagramGraph"
import { nodeSketchPaths, roughSeed } from "@/features/editor/architecture/architectureSketch"
import { cn } from "@/lib/utils"
import { StyledSelect } from "@/components/ui/StyledSelect"
import { nodeMinWidth } from "./architectureLabelWidth"
import { FALLBACK_SIZE, GRID_SIZE, NODE_MIN_HEIGHT, snapToGrid, type ArchitectureNodeData, type GraphFlowNode } from "./architectureFlow"

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
  const [titleDraft, setTitleDraft] = useState(data.title)
  const refreshTitleDraft = useEffectEvent((title: string) => {
    if (cardRef.current?.contains(document.activeElement)) return
    setTitleDraft(title)
  })
  useLayoutEffect(() => refreshTitleDraft(data.title), [data.title])
  const [bulletDrafts, setBulletDrafts] = useState<string[]>(() => [...data.bullets])
  const refreshBulletDrafts = useEffectEvent((bullets: readonly string[]) => {
    if (cardRef.current?.contains(document.activeElement)) return
    setBulletDrafts([...bullets])
  })
  useLayoutEffect(() => refreshBulletDrafts(data.bullets), [data.bullets])
  // Drop refs for removed rows so the autosize pass cannot touch an unmounted textarea.
  const bulletCount = bulletDrafts.length
  useLayoutEffect(() => { bulletRefs.current.length = bulletCount }, [bulletCount])
  const [pendingFocus, setPendingFocus] = useState<PendingFocus | null>(null)
  useLayoutEffect(() => {
    if (!pendingFocus) return
    const el = bulletRefs.current[pendingFocus.index]
    el?.focus()
    el?.setSelectionRange(pendingFocus.cursor, pendingFocus.cursor)
  }, [pendingFocus])
  const focusBullet = (index: number, cursor: number) => setPendingFocus({ index, cursor })

  // The commit reads the drafts through refs: a key handler may change them and
  // blur in the same event, before React re-renders.
  const titleDraftRef = useRef(titleDraft)
  const bulletDraftsRef = useRef(bulletDrafts)
  useLayoutEffect(() => {
    titleDraftRef.current = titleDraft
    bulletDraftsRef.current = bulletDrafts
  }, [titleDraft, bulletDrafts])
  const setBullets = (next: string[]) => {
    bulletDraftsRef.current = next
    setBulletDrafts(next)
  }
  const handleCardBlur = (event: FocusEvent<HTMLDivElement>) => {
    if (event.currentTarget.contains(event.relatedTarget as globalThis.Node | null)) return
    const title = titleDraftRef.current
    // Strip a bullet glyph or dash the user typed or pasted at the line start.
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

  const claimTitleEdit = useEffectEvent(() => {
    if (data.claimAutoEdit?.(data.id)) setEditingTitle(true)
  })
  // Mount only: a freshly created node claims its one-time edit, nothing else does.
  // eslint-disable-next-line react-hooks/set-state-in-effect -- claiming consumes the surface's one-shot token, which must follow the commit rather than a render that may be discarded
  useLayoutEffect(() => claimTitleEdit(), [])
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

export function ArchitectureNodeCard({ data, selected }: NodeProps<GraphFlowNode>) {
  const [hovered, setHovered] = useState(false)
  const cardRef = useRef<HTMLDivElement>(null)
  const [cardSize, setCardSize] = useState(FALLBACK_SIZE)
  // Track the rendered size so the sketch outline always matches the card.
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
  const { id: nodeId, width: nodeWidth, height: nodeHeight, onChange: changeNode } = data
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
    if (minNodeWidth > nodeWidth) changeNode(nodeId, { width: minNodeWidth })
  }, [minNodeWidth, nodeWidth, nodeId, changeNode])
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
      const nextWidth = overflowX > 0 ? Math.ceil((nodeWidth + overflowX) / GRID_SIZE) * GRID_SIZE : nodeWidth
      const growHeight = fit > nodeHeight
      if (!growHeight && nextWidth === nodeWidth) return
      changeNode(nodeId, { ...(growHeight ? { height: fit } : {}), ...(nextWidth !== nodeWidth ? { width: nextWidth } : {}) })
    })
    return () => cancelAnimationFrame(id)
  }, [bulletDrafts, titleDraft, editingTitle, cardSize, nodeHeight, nodeWidth, nodeId, changeNode, bulletRefs])
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
