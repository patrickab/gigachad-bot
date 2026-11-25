// The board's tree of planned work. Consumes the factory's waves (`waves.list`), provides `treeOfWaves`: tracks and
// milestones as the prefixes of the waves' tags, each with the sums of its waves, and the waves at their prefix.
import type { FactoryUsage, FactoryWave } from "@/lib/domain/factory"

/** One prefix of the waves' tags: `plugin` and `plugin/m1` of `plugin/m1/w1`, the root the empty prefix. */
export interface WaveGroup {
  name: string
  /** The tag prefix down to this segment. */
  path: string
  children: WaveGroup[]
  /** The waves whose tag is `path` plus one segment. */
  waves: FactoryWave[]
  /** Summed over every wave in this group and below. */
  counts: { total: number; done: number }
  /** Summed over every wave in this group and below that has usage, null when none has. */
  usage: FactoryUsage | null
  /** The memory_store of the group's plan: the first one a wave in it names, null when none does. */
  plan: string | null
}

/** The last segment of a wave's tag: its name in the tree. */
export function waveName(wave: FactoryWave): string {
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
  let counts = { total: 0, done: 0 }
  let usage: FactoryUsage | null = null
  for (const part of [...waves.map((wave) => ({ counts: wave.counts, usage: wave.usage })), ...children]) {
    counts = { total: counts.total + part.counts.total, done: counts.done + part.counts.done }
    usage = addUsage(usage, part.usage)
  }
  const plan = waves.find((wave) => wave.plan != null)?.plan ?? children.find((child) => child.plan != null)?.plan ?? null
  return { ...group, children, waves, counts, usage, plan }
}

/**
 * The waves as a tree: a group per tag prefix, nested, each wave in the group of its prefix (a one-segment tag at the
 * root). Groups and waves sort by name, numbers by value (`w2` before `w10`).
 */
export function treeOfWaves(waves: readonly FactoryWave[]): WaveGroup {
  const root: WaveGroup = { name: "", path: "", children: [], waves: [], counts: { total: 0, done: 0 }, usage: null, plan: null }
  for (const wave of waves) {
    let group = root
    const prefix = wave.tag.split("/").slice(0, -1)
    prefix.forEach((name, index) => {
      let child = group.children.find((candidate) => candidate.name === name)
      if (!child) {
        child = { name, path: prefix.slice(0, index + 1).join("/"), children: [], waves: [], counts: { total: 0, done: 0 }, usage: null, plan: null }
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
