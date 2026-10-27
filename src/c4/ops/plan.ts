// One planner per operation: what source edits and position changes it makes.
import type { PlanContext } from './context.ts'
import { OperationError, type Op, type Operation, type OperationName, type Plan } from './operation.ts'
import { patchPins } from './pins.ts'
import { deleteElements, pruneDangling } from './plan-delete.ts'
import { addElement, addRelation, setDescription, setKind, setLabel, setTitle } from './plan-elements.ts'
import { addFileView, createModule, createSystem, createView } from './plan-files.ts'
import { rename, reparent } from './plan-move.ts'
import { includeInView, removeFromView } from './plan-views.ts'

type Planner<K extends OperationName> = (c: PlanContext, op: Op<K>) => Plan | Promise<Plan>

const planners: { [K in OperationName]: Planner<K> } = {
  createSystem,
  createModule,
  createView,
  addFileView,
  addElement,
  addRelation,
  delete: deleteElements,
  pruneDangling,
  setTitle,
  setDescription,
  setKind,
  setLabel,
  rename,
  reparent,
  includeInView,
  removeFromView,
  // Positions are saved, not written into source.
  layout: (_c, op) => ({ edits: [], repin: (pins) => patchPins(pins, op) }),
}

export async function plan(c: PlanContext, op: Operation): Promise<Plan> {
  // Requests are unchecked JSON, so `op.op` may name no planner at all.
  const name: unknown = op.op
  if (typeof name !== 'string' || !Object.hasOwn(planners, name)) throw new OperationError(`Unknown operation: ${name}`)
  // TypeScript cannot correlate `op` with `planners[op.op]`; the mapped type guarantees they match.
  return (planners[op.op] as Planner<OperationName>)(c, op)
}
