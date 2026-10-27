// The structured edits the UI requests, and what planning one produces.
import type { NodePin, Pins, Snapshots } from '../layout.ts'
import type { ModelPayload } from '../payload.ts'
import type { Sources } from '../workspace.ts'
import type { TextEdit } from './text.ts'

export type Operation =
  | { op: 'createSystem', title: string }
  | { op: 'createModule', system: string, title: string }
  | { op: 'createView', title: string }
  | { op: 'addFileView' }
  | { op: 'addElement', parent: string | null, kind: string, title: string, description?: string, layout?: { view: string } & NodePin }
  | { op: 'addRelation', source: string, target: string, label?: string }
  | { op: 'delete', elements: string[], relations: string[] }
  | { op: 'pruneDangling' }
  | { op: 'setTitle', element: string, title: string }
  | { op: 'setDescription', element: string, description: string }
  | { op: 'setKind', element: string, kind: string }
  | { op: 'setLabel', relation: string, label: string }
  | { op: 'rename', element: string, id: string }
  | { op: 'reparent', element: string, parent: string | null }
  | { op: 'includeInView', view: string, elements: string[], descendants?: boolean }
  | { op: 'removeFromView', view: string, elements: string[] }
  | { op: 'layout', view: string, nodes: Record<string, NodePin | null>, edges: Record<string, { bend: number } | null> }

export type OperationName = Operation['op']

/** The operation named `K`. */
export type Op<K extends OperationName> = Extract<Operation, { op: K }>

export interface ApplyResult {
  sources: Sources
  /** Every snapshot file the workspace should now hold; any other `.likec4/*.snap` is stale. */
  snapshots: Snapshots
  model: ModelPayload
  /**
   * Per operation: the FQN an `addElement` created, the path a `createSystem`,
   * `createModule` or `createView` created, the view id an `addFileView` declared, otherwise null.
   */
  created: Array<string | null>
}

/** A request the current model cannot honour; the message is shown to the user. */
export class OperationError extends Error {}

/** What one operation does: source edits, plus how saved positions change. */
export interface Plan {
  edits: TextEdit[]
  /** Old FQN -> new FQN, for saved positions and the equivalence check. */
  renames?: Record<string, string>
  /** Saved-position changes beyond following `renames`. */
  repin?: (pins: Pins) => Pins
  /** The FQN, file path or view id reported back in `ApplyResult.created`. */
  created?: string
}
