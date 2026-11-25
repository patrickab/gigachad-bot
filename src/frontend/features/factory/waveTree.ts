// The board's waves. Consumes the factory's waves (`waves.list`), provides the column a wave's container sits in
// (`waveColumn`) and `treeOfWaves`: the waves of one column nested in their tags' prefixes (tracks and milestones), each
// prefix with the count of its waves, the merged ones and their summed usage.
import type { FactoryUsage, FactoryWave } from "@/lib/domain/factory"

/** The board columns a wave's container sits in: Working holds only tasks. */
export type WaveColumn = "waiting" | "merging" | "finished"

/**
 * The column of a wave's container, null while it has none: a building wave shows only once a task of it is done.
 * Interrupted and resting on the supervisor (`rejected`, `blocked`, `waiting_approve`) waits, being built, reviewed or
 * merged is the wave in Merging, `merged` and `cancelled` are finished.
 */
export function waveColumn(wave: Pick<FactoryWave, "stage" | "interrupted">, hasDone: boolean): WaveColumn | null {
  if (wave.interrupted) return "waiting"
  switch (wave.stage) {
    case "rejected":
    case "blocked":
    case "waiting_approve":
      return "waiting"
    case "merged":
    case "cancelled":
      return "finished"
    case "building":
      return hasDone ? "merging" : null
    default:
      return "merging"
  }
}

/** One prefix of the waves' tags: `plugin` and `plugin/m1` of `plugin/m1/w1`, the root the empty prefix. */
export interface WaveGroup {
  name: string
  /** The tag prefix down to this segment. */
  path: string
  children: WaveGroup[]
  /** The waves whose tag is `path` plus one segment. */
  waves: FactoryWave[]
  /** The waves in this group and below, and the merged ones of them. */
  counts: { total: number; merged: number }
  /** Summed over every wave in this group and below that has usage, null when none has. */
  usage: FactoryUsage | null
  /** The memory_store of the group's plan: the first one a wave in it names, null when none does. */
  plan: string | null
}

/** The last segment of a wave's tag: its name on its tasks' cards. */
export function waveName(wave: Pick<FactoryWave, "tag">): string {
  return wave.tag.split("/").at(-1) || wave.tag
}

const byName = (left: string, right: string) => left.localeCompare(right, undefined, { numeric: true })

function addUsage(sum: FactoryUsage | null, usage: FactoryUsage | null): FactoryUsage | null {
  if (!usage) return sum
  return { input: (sum?.input ?? 0) + usage.input, output: (sum?.output ?? 0) + usage.output, cost: (sum?.cost ?? 0) + usage.cost }
}

function finish(group: WaveGroup): WaveGroup {
  const children = group.children.map(finish).sort((left, right) => byName(left.name, right.name))
  const waves = [...group.waves].sort((left, right) => byName(waveName(left), waveName(right)) || left.createdAt.localeCompare(right.createdAt))
  let counts = { total: waves.length, merged: waves.filter((wave) => wave.stage === "merged").length }
  let usage: FactoryUsage | null = null
  for (const wave of waves) usage = addUsage(usage, wave.usage)
  for (const child of children) {
    counts = { total: counts.total + child.counts.total, merged: counts.merged + child.counts.merged }
    usage = addUsage(usage, child.usage)
  }
  const plan = waves.find((wave) => wave.plan != null)?.plan ?? children.find((child) => child.plan != null)?.plan ?? null
  return { ...group, children, waves, counts, usage, plan }
}

/**
 * The waves as a tree: a group per tag prefix, nested, each wave in the group of its prefix (a one-segment tag at the
 * root). Groups and waves sort by name, numbers by value (`w2` before `w10`).
 */
export function treeOfWaves(waves: readonly FactoryWave[]): WaveGroup {
  const root: WaveGroup = { name: "", path: "", children: [], waves: [], counts: { total: 0, merged: 0 }, usage: null, plan: null }
  for (const wave of waves) {
    let group = root
    const prefix = wave.tag.split("/").slice(0, -1)
    prefix.forEach((name, index) => {
      let child = group.children.find((candidate) => candidate.name === name)
      if (!child) {
        child = { name, path: prefix.slice(0, index + 1).join("/"), children: [], waves: [], counts: { total: 0, merged: 0 }, usage: null, plan: null }
        group.children.push(child)
      }
      group = child
    })
    group.waves.push(wave)
  }
  return finish(root)
}

/** The memory_store document describing the track, milestone or wave at tag `path`. */
export function descriptionPath(path: string): string {
  return `plan/${path}.md`
}

/** A description's title: its first non-empty line without markdown heading marks, null when it has none. */
export function descriptionTitle(text: string): string | null {
  const line = text.split("\n").map((row) => row.replace(/^\s*#+\s*/, "").trim()).find(Boolean)
  return line ?? null
}
