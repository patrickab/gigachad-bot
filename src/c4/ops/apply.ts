// Applies a batch of operations all or nothing, checking each edited model
// before moving on and carrying saved positions along.
import { mapPrefix } from '../fqn.ts'
import type { Pins } from '../layout.ts'
import { renderModel } from '../payload.ts'
import { Workspace, type Sources } from '../workspace.ts'
import { PlanContext } from './context.ts'
import { OperationError, type ApplyResult, type Operation } from './operation.ts'
import { renamePins } from './pins.ts'
import { plan } from './plan.ts'
import { followUp } from './plan-views.ts'
import { applyEdits, type TextEdit } from './text.ts'

const counts = (values: string[]) => {
  const result = new Map<string, number>()
  for (const value of values) result.set(value, (result.get(value) ?? 0) + 1)
  return result
}

/** Errors the edit introduced; pre-existing breakage elsewhere doesn't block unrelated edits. */
function newErrors(before: Workspace, after: Workspace): string[] {
  const known = counts(before.errors.map((error) => error.message))
  const introduced: string[] = []
  for (const error of after.errors) {
    const left = known.get(error.message) ?? 0
    if (left > 0) known.set(error.message, left - 1)
    else introduced.push(`${error.file}:${error.line + 1}: ${error.message}`)
  }
  return introduced
}

/** A rename or move must reproduce the same model under the FQN mapping. */
async function assertEquivalent(before: Workspace, after: Workspace, renames: Record<string, string>): Promise<void> {
  const shape = async (ws: Workspace, map: (fqn: string) => string) => {
    const model = await ws.model()
    return [
      ...[...model.elements()].map((element) => `element ${map(element.id)} ${element.kind}`),
      ...[...model.relationships()].map((relation) => `relation ${map(relation.source.id)} ${map(relation.target.id)} ${relation.title ?? ''}`),
    ].sort().join('\n')
  }
  if (await shape(before, (fqn) => mapPrefix(fqn, renames)) !== await shape(after, (fqn) => fqn)) {
    throw new OperationError('Could not rewrite every reference safely; nothing was changed')
  }
}

/** The workspace with `edits` applied, once it loads without new errors; disposes `workspace` on success. */
async function commit(workspace: Workspace, edits: TextEdit[], renames: Record<string, string>): Promise<Workspace> {
  const next = await Workspace.open(applyEdits(workspace.sources, edits)).catch((error: unknown) => {
    throw new OperationError(`LikeC4 cannot load the edited model: ${error instanceof Error ? error.message : String(error)}`)
  })
  try {
    const introduced = newErrors(workspace, next)
    if (introduced.length > 0) throw new OperationError(introduced[0]!)
    if (Object.keys(renames).length > 0) await assertEquivalent(workspace, next, renames)
  } catch (error) {
    await next.dispose()
    throw error
  }
  await workspace.dispose()
  return next
}

interface State {
  workspace: Workspace
  pins: Pins
}

/** Plans and commits one operation, returning what it created. */
async function step(state: State, operation: Operation, home: string | null): Promise<string | null> {
  const { edits, renames = {}, repin, created = null } = await plan(new PlanContext(state.workspace, home), operation)
  if (edits.length > 0) state.workspace = await commit(state.workspace, edits, renames)
  state.pins = renamePins(state.pins, renames)
  if (repin) state.pins = repin(state.pins)
  return created
}

/**
 * Applies `operations` in order, all or nothing. `home` is the file new
 * top-level elements are written into (the window's file). Saved positions
 * (`pins`) follow renames, take `layout` operations, and come back as the
 * snapshot files of every pinned view.
 */
export async function applyOperations(sources: Sources, pins: Pins, operations: Operation[], home: string | null): Promise<ApplyResult> {
  const state: State = { workspace: await Workspace.open(sources), pins }
  const created: Array<string | null> = []
  try {
    for (const operation of operations) {
      const made = await step(state, operation, home)
      created.push(made)
      // A follow-up runs right after the operation it completes and reports nothing itself.
      const extra = await followUp(state.workspace, operation, made)
      if (extra) await step(state, extra, home)
    }
    const { model, snapshots } = await renderModel(state.workspace, state.pins)
    return { sources: state.workspace.sources, snapshots, model, created }
  } finally {
    await state.workspace.dispose()
  }
}
