// Saved positions: following renames and applying `layout` patches.
import { mapPrefix } from '../fqn.ts'
import { edgeKey, splitEdgeKey, type Pins, type ViewPins } from '../layout.ts'
import type { Op } from './operation.ts'

/** Saved positions follow renamed and moved elements. */
export function renamePins(pins: Pins, renames: Record<string, string>): Pins {
  if (Object.keys(renames).length === 0) return pins
  return Object.fromEntries(Object.entries(pins).map(([viewId, view]) => [viewId, {
    nodes: Object.fromEntries(Object.entries(view.nodes).map(([fqn, pin]) => [mapPrefix(fqn, renames), pin])),
    edges: Object.fromEntries(Object.entries(view.edges).map(([key, bend]) => {
      const [source, target] = splitEdgeKey(key)
      return [edgeKey(mapPrefix(source, renames), mapPrefix(target, renames)), bend]
    })),
  }]))
}

/** A null entry unpins: the node returns to LikeC4's placement, the connection straightens (a zero bend does too). */
export function patchPins(pins: Pins, op: Op<'layout'>): Pins {
  const view: ViewPins = { nodes: { ...pins[op.view]?.nodes }, edges: { ...pins[op.view]?.edges } }
  for (const [fqn, pin] of Object.entries(op.nodes)) {
    if (pin) view.nodes[fqn] = pin
    else delete view.nodes[fqn]
  }
  for (const [key, path] of Object.entries(op.edges)) {
    if (path && path.bend) view.edges[key] = path.bend
    else delete view.edges[key]
  }
  return { ...pins, [op.view]: view }
}
