// Planners that create whole files: systems, modules, saved views, a file's view.
import { nameOf, parentOf } from '../fqn.ts'
import type { PlanContext } from './context.ts'
import { idFromTitle, titleFromName, uniqueId } from './ids.ts'
import { OperationError, type Op, type Plan } from './operation.ts'
import { appendBlockAtEof, block, INDENT_UNIT, insertAt, quote, requireTitle } from './text.ts'

/** Saved views live one per file under this folder; the file name is the view id. */
const VIEWS_DIR = 'views/'

/** Element kinds a workspace starts with, written into its first system. */
const DEFAULT_SPECIFICATION = `specification {
  element actor {
    style {
      shape person
    }
  }
  element system
  element container
  element component
  element database {
    style {
      shape cylinder
    }
  }
}

`

/** The file a new system, module or view gets: `id` made unique against `taken` names and existing files. */
function newFile(c: PlanContext, folder: string, title: string, taken: ReadonlySet<string>): { id: string, path: string } {
  const id = uniqueId(idFromTitle(title), new Set([...taken, ...Object.keys(c.ws.sources)
    .filter((path) => path.startsWith(folder) && !path.slice(folder.length).includes('/'))
    .map((path) => path.slice(folder.length, -'.c4'.length))]))
  return { id, path: `${folder}${id}.c4` }
}

async function kindFor(c: PlanContext, preferred: string): Promise<string> {
  const kinds = await c.kinds()
  return kinds.size === 0 || kinds.has(preferred) ? preferred : [...kinds][0]!
}

/** A system is a folder, `<id>/<id>.c4`, declaring one system and the view of it. */
export async function createSystem(c: PlanContext, op: Op<'createSystem'>): Promise<Plan> {
  const title = requireTitle(op.title, 'A system needs a name')
  // Neither an element nor a folder may already use the id, so `views/` is never a system.
  const folders = Object.keys(c.ws.sources).filter((path) => path.includes('/')).map((path) => path.slice(0, path.indexOf('/')))
  const id = uniqueId(idFromTitle(title), new Set([...await c.childNames(null), ...folders, VIEWS_DIR.slice(0, -1)]))
  const path = `${id}/${id}.c4`
  const specification = (await c.kinds()).size === 0 ? DEFAULT_SPECIFICATION : ''
  const model = block('model', [`${id} = ${await kindFor(c, 'system')} ${quote(title)}`])
  const views = block('views', [`view ${uniqueId(id, c.ws.viewIds())} of ${id} {`, `${INDENT_UNIT}include *`, '}'])
  return { edits: [insertAt(path, 0, `${specification}${model}\n${views}`)], created: path }
}

/** A module is a file beside its system's, adding one container to it with `extend`. */
export async function createModule(c: PlanContext, op: Op<'createModule'>): Promise<Plan> {
  const title = requireTitle(op.title, 'A module needs a name')
  const system = c.element(op.system)
  if (parentOf(op.system) !== null) throw new OperationError('Modules belong to a top-level system')
  const home = c.ws.documentPath(system)
  const folder = home.includes('/') ? home.slice(0, home.lastIndexOf('/') + 1) : `${op.system}/`
  const { id, path } = newFile(c, folder, title, await c.childNames(op.system))
  const text = block('model', [`extend ${op.system} {`, `${INDENT_UNIT}${id} = ${await kindFor(c, 'container')} ${quote(title)}`, '}'])
  return { edits: [insertAt(path, 0, text)], created: path }
}

/** A saved view is its own file, `views/<id>.c4`, and starts empty. */
export function createView(c: PlanContext, op: Op<'createView'>): Plan {
  const title = requireTitle(op.title, 'A view needs a name')
  const { id, path } = newFile(c, VIEWS_DIR, title, c.ws.viewIds())
  const text = block('views', [`view ${id} {`, `${INDENT_UNIT}title ${quote(title)}`, '}'])
  return { edits: [insertAt(path, 0, text)], created: path }
}

/** Gives a file without a view one: a view of its single system, else a list of what it declares. */
export function addFileView(c: PlanContext): Plan {
  const file = c.homeFile()
  const roots = c.ws.rootsOf(file)
  const base = file.slice(file.lastIndexOf('/') + 1, -'.c4'.length)
  const single = roots.length === 1 ? roots[0]! : null
  const id = uniqueId(single ? nameOf(single) : idFromTitle(titleFromName(base)), c.ws.viewIds())
  const body = single ? [`view ${id} of ${single} {`, `${INDENT_UNIT}include *`, '}'] : [`view ${id} {`, ...(roots.length ? [`${INDENT_UNIT}include ${roots.join(', ')}`] : []), '}']
  return { edits: [appendBlockAtEof(file, c.text(file), 'views', body, 1)], created: id }
}
