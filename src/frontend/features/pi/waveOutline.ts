// Waves as one tree, a line per row (a Pi tab's `SessionWaves`). Consumes the core's waves, provides `outlineOfWaves`:
// the summary (spend, workers, active or finished waves) and the rows, each with its tree lines, mark, label, title
// and facts.
//
// - The tags' prefixes nest: `plugin/m1/w1` is wave `w1` in milestone `m1` of track `plugin`. Groups and waves sort by
//   name, numbers by value (`w2` before `w10`)
//
// - A group holding one wave is not drawn: its name joins the wave's label
// - In a group: what needs the user first; what is finished (`merged`, `cancelled`, a group of only those) under one
//   `<n> finished` row
// - Open unless toggled: groups, and waves that are neither queued nor finished. The `finished` rows start closed
import { waveNeedsUser, waveStageLabel, type FactoryWave, type FactoryWaveTask } from "@/lib/domain/factory"
import { formatCost } from "./subagents"

export type OutlineMark = "working" | "done" | "cancelled" | "attention" | "failed" | "waiting"
export type OutlineTone = "ink" | "muted" | "faint" | "warning" | "danger"
export interface OutlineFact { text: string; tone: OutlineTone }

export interface OutlineRow {
  key: string
  /** What a click toggles, null for a row with nothing below. */
  toggle: string | null
  /** The task a click opens, null for every other row and a task not started. */
  task: { repo: string; taskId: string } | null
  /** The tree lines before the row. */
  lead: string
  open: boolean | null
  mark: OutlineMark | null
  label: string
  /** A track or milestone: its label carries weight. */
  group: boolean
  title: string
  facts: OutlineFact[]
}

export interface Outline { summary: OutlineFact[]; rows: OutlineRow[] }

const finished = (wave: FactoryWave) => wave.stage === "merged" || wave.stage === "cancelled"
const failed = (wave: FactoryWave) => wave.interrupted || wave.stage === "blocked"

/** The tasks a coder runs for, and the wave itself while its reviewer runs. */
function workers(wave: FactoryWave): number {
  return wave.tasks.filter((task) => task.state === "running").length + Number(wave.running && wave.stage === "reviewing")
}

function sum(waves: readonly FactoryWave[], pick: (wave: FactoryWave) => number): number {
  return waves.reduce((total, wave) => total + pick(wave), 0)
}

function cost(waves: readonly FactoryWave[]): string {
  return waves.some((wave) => wave.usage) ? formatCost(sum(waves, (wave) => wave.usage?.cost ?? 0)) : ""
}

function count(n: number, one: string): string {
  return `${n} ${one}${n === 1 ? "" : "s"}`
}

/** `42s`, `8m`, `1h05m`. */
function duration(ms: number): string {
  const minutes = Math.floor(ms / 60_000)
  if (minutes < 1) return `${Math.max(0, Math.floor(ms / 1000))}s`
  return minutes < 60 ? `${minutes}m` : `${Math.floor(minutes / 60)}h${String(minutes % 60).padStart(2, "0")}m`
}

/** How long the wave took until it merged, or has taken until `now` once it started; nothing for a cancelled one. */
function elapsed(wave: FactoryWave, now: number): string {
  const end = wave.merge ? Date.parse(wave.merge.at) : finished(wave) || wave.stage === "queued" ? Number.NaN : now
  const ms = end - Date.parse(wave.createdAt)
  return Number.isFinite(ms) && ms >= 0 ? duration(ms) : ""
}

function fact(text: string, tone: OutlineTone = "faint"): OutlineFact[] {
  return text ? [{ text, tone }] : []
}

function waveMark(wave: FactoryWave): OutlineMark {
  if (wave.stage === "merged") return "done"
  if (wave.stage === "cancelled") return "cancelled"
  if (failed(wave)) return "failed"
  if (waveNeedsUser(wave)) return "attention"
  return wave.stage === "queued" ? "waiting" : "working"
}

function taskMark(task: FactoryWaveTask): OutlineMark {
  switch (task.state) {
    case "done": return "done"
    case "cancelled": return "cancelled"
    case "pending": return "waiting"
    case "blocked": return "failed"
    case "waiting_answer":
    case "waiting_approve": return "attention"
    default: return "working"
  }
}

const MARK_TONE: Record<OutlineMark, OutlineTone> = { working: "muted", done: "faint", cancelled: "faint", attention: "warning", failed: "danger", waiting: "faint" }

/** One prefix of the waves' tags with what lies under it. */
interface Group { path: string; groups: Map<string, Group>; waves: FactoryWave[] }
/** A wave with the label it shows under, or a group of them. */
type Entry = { label: string; wave: FactoryWave } | { label: string; group: Group }

const waveName = (wave: FactoryWave) => wave.tag.split("/").at(-1) || wave.tag

function treeOfWaves(waves: readonly FactoryWave[]): Group {
  const root: Group = { path: "", groups: new Map(), waves: [] }
  for (const wave of waves) {
    let group = root
    for (const name of wave.tag.split("/").slice(0, -1)) {
      const child = group.groups.get(name) ?? { path: group.path ? `${group.path}/${name}` : name, groups: new Map(), waves: [] }
      group.groups.set(name, child)
      group = child
    }
    group.waves.push(wave)
  }
  return root
}

/** What a group holds, by name; a group holding one wave is that wave, its name joined to the wave's. */
function entriesOf(group: Group): Entry[] {
  const entries = [
    ...[...group.groups].map(([name, child]): Entry => {
      const only = child.groups.size === 0 && child.waves.length === 1 ? child.waves[0] : null
      return only ? { label: `${name}/${waveName(only)}`, wave: only } : { label: name, group: child }
    }),
    ...group.waves.map((wave): Entry => ({ label: waveName(wave), wave })),
  ]
  return entries.sort((left, right) => left.label.localeCompare(right.label, undefined, { numeric: true }))
}

function wavesOf(entry: Entry): FactoryWave[] {
  return "wave" in entry ? [entry.wave] : entriesOf(entry.group).flatMap(wavesOf)
}

/**
 * `waves` as an outline. `toggled`: the rows whose open state is the opposite of their default; `models`: the coder
 * model of a task by task id.
 */
export function outlineOfWaves(waves: readonly FactoryWave[], toggled: ReadonlySet<string>, now: number, models: Readonly<Record<string, string>>): Outline {
  const rows: OutlineRow[] = []
  const isOpen = (id: string, byDefault: boolean) => byDefault !== toggled.has(id)
  const byId = new Map(waves.map((wave) => [wave.id, wave]))

  const waveRows = (label: string, wave: FactoryWave, lead: string, below: string) => {
    const id = `wave:${wave.id}`
    const open = wave.tasks.length === 0 ? null : isOpen(id, !finished(wave) && wave.stage !== "queued")
    const mark = waveMark(wave)
    const busy = workers(wave)
    const rejections = wave.reviews.filter((review) => review.verdict === "rejected").length
    const after = wave.stage === "queued" ? (wave.after ?? []).map((other) => byId.get(other)?.tag ?? other.slice(0, 8)).join(", ") : ""
    rows.push({
      key: id, toggle: open === null ? null : id, task: null, lead, open, mark, label, group: false, title: wave.title,
      facts: [
        ...fact(`${wave.counts.done}/${wave.counts.total} tasks`),
        ...fact(waveStageLabel(wave), MARK_TONE[mark]),
        ...fact(after && `after ${after}`),
        ...fact(busy > 0 ? count(busy, "worker") : "", "muted"),
        ...fact(rejections > 0 ? count(rejections, "rejection") : ""),
        ...fact(elapsed(wave, now)),
        ...fact(cost([wave])),
      ],
    })
    if (!open) return
    wave.tasks.forEach((task, index) => {
      const taskState = taskMark(task)
      rows.push({
        key: `${id}/${task.key}`, toggle: null, task: task.taskId ? { repo: wave.repo, taskId: task.taskId } : null, lead: `${below}${index === wave.tasks.length - 1 ? "└─ " : "├─ "}`, open: null, mark: taskState,
        label: task.key, group: false, title: task.title,
        facts: [
          ...fact(task.taskId ? models[task.taskId] ?? "" : ""),
          ...fact(task.state.replace(/_/g, " "), MARK_TONE[taskState]),
          ...fact(task.state === "pending" && task.after.length > 0 ? `after ${task.after.join(", ")}` : ""),
          ...fact(task.usage ? formatCost(task.usage.cost) : ""),
        ],
      })
    })
  }

  /** `fold`: what is finished goes under one `<n> finished` row. */
  const level = (entries: Entry[], indent: string, fold: boolean, path: string) => {
    const done = (entry: Entry) => wavesOf(entry).every(finished)
    const urgent = (entry: Entry) => Number(wavesOf(entry).some(waveNeedsUser))
    // `sort` is stable: within the urgent and the rest the tree's order by name stays
    const shown = entries.filter((entry) => !fold || !done(entry)).sort((left, right) => urgent(right) - urgent(left))
    const rest = fold ? entries.filter(done) : []
    shown.forEach((entry, index) => {
      const last = index === shown.length - 1 && rest.length === 0
      const lead = `${indent}${last ? "└─ " : "├─ "}`
      const below = `${indent}${last ? "   " : "│  "}`
      if ("wave" in entry) return waveRows(entry.label, entry.wave, lead, below)
      const inside = wavesOf(entry)
      const id = `group:${entry.group.path}`
      const open = isOpen(id, true)
      const busy = sum(inside, workers)
      const mark: OutlineMark = inside.some(failed) ? "failed" : inside.some(waveNeedsUser) ? "attention" : inside.every(finished) ? "done" : inside.every((wave) => finished(wave) || wave.stage === "queued") ? "waiting" : "working"
      rows.push({
        key: id, toggle: id, task: null, lead, open, mark, label: entry.label, group: true, title: "",
        facts: [...fact(`${inside.filter((wave) => wave.stage === "merged").length}/${inside.length} waves`), ...fact(busy > 0 ? count(busy, "worker") : "", "muted"), ...fact(cost(inside))],
      })
      if (open) level(entriesOf(entry.group), below, fold, `${entry.group.path}/`)
    })
    if (rest.length === 0) return
    const inside = rest.flatMap(wavesOf)
    const id = `finished:${path}`
    const open = isOpen(id, false)
    rows.push({ key: id, toggle: id, task: null, lead: `${indent}└─ `, open, mark: null, label: `${inside.length} finished`, group: false, title: "", facts: fact(cost(inside)) })
    if (open) level(rest, `${indent}   `, false, `${id}/`)
  }
  level(entriesOf(treeOfWaves(waves)), "", true, "")

  const active = waves.filter((wave) => !finished(wave))
  const busy = sum(waves, workers)
  return {
    rows,
    summary: [
      ...fact(cost(waves), "ink"),
      ...fact(busy > 0 ? count(busy, "worker") : "", "muted"),
      ...fact(active.length > 0 ? `${active.length} active` : `${waves.length} finished`),
    ],
  }
}
