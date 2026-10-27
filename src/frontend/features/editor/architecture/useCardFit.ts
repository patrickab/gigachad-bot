import { useEffect, useRef, useState, type RefObject } from "react"
import { nodeMinWidth } from "./architectureLabelWidth"
import { GRID_SIZE, NODE_MIN_HEIGHT, type ArchitectureNodeData } from "./architectureFlow"

/**
 * Auto-fit: the content's natural size is a floor for the card. The card
 * grows when text stops fitting (vertically or horizontally) and never
 * shrinks on its own, so a size the user chose sticks until the text needs more.
 * Also autosizes the bullet textareas. `fitHeight` and `minNodeWidth` are the resize floors;
 * while `resizingRef` is set the card only measures, so a resize drag is not fought.
 */
export function useCardFit(data: ArchitectureNodeData, { titleDraft, bulletDrafts, editingTitle, bulletRefs, cardSize }: {
  titleDraft: string
  bulletDrafts: readonly string[]
  editingTitle: boolean
  bulletRefs: RefObject<Array<HTMLTextAreaElement | null>>
  cardSize: { width: number, height: number }
}) {
  const { id: nodeId, width: nodeWidth, height: nodeHeight, onChange: changeNode } = data
  // A title bar wider than the card would truncate the heading and push the kind
  // label past the edge, so their width is a resize floor (and grows the card while typing).
  const minNodeWidth = nodeMinWidth(titleDraft, data.kind)
  const titlebarRef = useRef<HTMLDivElement>(null)
  const bodyRef = useRef<HTMLDivElement>(null)
  const resizingRef = useRef(false)
  const [fitHeight, setFitHeight] = useState(NODE_MIN_HEIGHT)
  useEffect(() => {
    if (minNodeWidth > nodeWidth) changeNode(nodeId, { width: minNodeWidth })
  }, [minNodeWidth, nodeWidth, nodeId, changeNode])
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
  return { titlebarRef, bodyRef, resizingRef, fitHeight, minNodeWidth }
}
