// The board's plan tree. Consumes the tasks' `tag` (a login alone, or a plan path down to its wave), provides
// `groupByTag`: the tasks of plan paths as a tree of groups, the others for the state columns.
import type { FactoryTask } from "@/lib/domain/factory"
import { STATE_ORDER, shownState, type FactoryShownState } from "./states"

/** One segment of a plan path: `plugin` and `m1` of `plugin/m1/w1`, the last one the wave. */
export interface TagGroup {
  name: string
  /** The tag prefix down to this segment. */
  path: string
  children: TagGroup[]
  /** Tasks tagged exactly with `path`: a wave's. */
  tasks: FactoryTask[]
  /** Tasks in this group and every group below it. */
  count: number
  /** How many of `tasks` are in each shown state, those with none left out, in board order. */
  states: { state: FactoryShownState; count: number }[]
  /** The memory_store of the group's plan: the first one a task in it names, null when none does. */
  plan: string | null
}

export interface TagGrouping {
  groups: TagGroup[]
  /** Tasks tagged with a single segment (a login or system user): they stay in the columns. */
  ungrouped: FactoryTask[]
}

/** Newest first, the id breaking ties, so a group's order does not depend on the order the core listed them in. */
function newestFirst(left: FactoryTask, right: FactoryTask): number {
  return right.createdAt.localeCompare(left.createdAt) || left.id.localeCompare(right.id)
}

const byName = (left: TagGroup, right: TagGroup) => left.name.localeCompare(right.name, undefined, { numeric: true })

function finish(group: TagGroup): TagGroup {
  const children = group.children.map(finish).sort(byName)
  const tasks = [...group.tasks].sort(newestFirst)
  const states = STATE_ORDER
    .map((state) => ({ state, count: tasks.filter((task) => shownState(task) === state).length }))
    .filter(({ count }) => count > 0)
  const plan = tasks.find((task) => task.plan !== null)?.plan ?? children.find((child) => child.plan !== null)?.plan ?? null
  return { ...group, children, tasks, states, plan, count: tasks.length + children.reduce((sum, child) => sum + child.count, 0) }
}

/**
 * Split `tasks` into the plan tree and the tasks for the columns.
 *
 * - A tag of two or more segments is a plan path: one group per segment, nested, the task in the group of its last one
 * - Groups sort by name (numbers by value: `w2` before `w10`), a group's tasks newest first
 */
export function groupByTag(tasks: readonly FactoryTask[]): TagGrouping {
  const roots: TagGroup[] = []
  const ungrouped: FactoryTask[] = []
  for (const task of tasks) {
    const segments = task.tag.split("/")
    if (segments.length < 2) {
      ungrouped.push(task)
      continue
    }
    let level = roots
    let group: TagGroup | undefined
    segments.forEach((name, index) => {
      group = level.find((candidate) => candidate.name === name)
      if (!group) {
        group = { name, path: segments.slice(0, index + 1).join("/"), children: [], tasks: [], count: 0, states: [], plan: null }
        level.push(group)
      }
      level = group.children
    })
    group?.tasks.push(task)
  }
  return { groups: roots.map(finish).sort(byName), ungrouped }
}

/** The memory_store document describing the group at `path`. */
export function descriptionPath(path: string): string {
  return `plan/${path}.md`
}
