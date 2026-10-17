import type { StrokeData } from "@/lib/drawing"

// Every mounted canvas registers its drawing surface here, so a selection dragged out of
// one canvas can be handed to whichever canvas is under the pointer on release — into a
// nested window or back out of one. The handoff is in client coordinates, so the ink
// lands exactly where it was dropped and comes out at the target's zoom: same on-screen
// size, which means `width` scales by the zoom ratio (a lasso resize deliberately leaves
// the pen alone; a canvas with a different zoom is the opposite case).

/** A canvas surface as the transfer sees it: where it sits on screen, and how it maps its own units onto that. `rect` only needs left/top, so tests can pass a bare object. */
export interface CanvasView { rect: { left: number; top: number }; scale: number; offset: { x: number; y: number } }
export type StrokeDrop = (strokes: StrokeData[], from: CanvasView) => void
const canvasDropTargets = new Map<HTMLElement, StrokeDrop>()

/** Register `el` as a drop surface; returns the unregister function. */
export function registerDropTarget(el: HTMLElement, drop: StrokeDrop): () => void {
  canvasDropTargets.set(el, drop)
  return () => { canvasDropTargets.delete(el) }
}

// Rewrite a stroke from one canvas's units into another's, going through client px, so
// it keeps both its position and its size on screen. `width` is in canvas units, so it
// takes the zoom ratio too — dropping into a canvas zoomed out 2× would otherwise double
// the apparent pen thickness.
export function remapAcrossCanvases(stroke: StrokeData, from: CanvasView, to: CanvasView): StrokeData {
  return {
    ...stroke,
    width: stroke.width * (from.scale / to.scale),
    points: stroke.points.map((p) => [
      (from.rect.left + p[0]! * from.scale + from.offset.x - to.rect.left - to.offset.x) / to.scale,
      (from.rect.top + p[1]! * from.scale + from.offset.y - to.rect.top - to.offset.y) / to.scale,
      p[2] ?? 0.5,
    ]),
  }
}

/** Innermost registered surface under the point, skipping the one being dragged from — walking up from the topmost element is what picks the nested window over its host. */
export function dropTargetAt(x: number, y: number, self: HTMLElement | null): StrokeDrop | null {
  let el: Element | null = document.elementFromPoint(x, y)
  while (el) {
    const drop = canvasDropTargets.get(el as HTMLElement)
    if (drop && el !== self) return drop
    el = el.parentElement
  }
  return null
}

// True when an event landed in an attachment window that *this* surface owns — those
// (PDFs, nested canvases) handle their own scroll and zoom, so the host must keep its
// hands off. Containment is the whole point: a nested canvas's own container also sits
// inside an attachment element, its own, and that one must still zoom itself. Native
// listeners run before React's synthetic dispatch, so a JSX stopPropagation can't do it.
export function inOwnAttachment(target: EventTarget | null, el: HTMLElement): boolean {
  const att = (target as Element | null)?.closest?.("[data-canvas-attachment]")
  return !!att && el.contains(att)
}
