/**
 * The subagent roster's data: what the user's pi-subagents extension exposes over Pi RPC, folded into one row per child.
 *
 * - Background children: the `subagent-async` text widget (`PI_SUBAGENT_ASYNC_JSON:` snapshot), carried by pi-host in the attach snapshot's
 *   `ui.widgets` and live `setWidget` records. Agent, state, current tool and times, no tokens or cost
 * - Foreground children: the `subagent` tool call's `details` (`progress[]` while running, `results[]` once done), with tokens and cost
 * - On demand: the `/subagents-inspect-rpc` reply (`PI_SUBAGENT_INSPECT_JSON:` on the transient `subagent-inspect` widget) for a background child's task, transcript and output
 * - Every field is optional and read leniently, unknown ones are ignored
 */
import type { ToolRun } from "./types"

export const ASYNC_WIDGET_KEY = "subagent-async"
export const INSPECT_WIDGET_KEY = "subagent-inspect"
const ASYNC_PREFIX = "PI_SUBAGENT_ASYNC_JSON:"
const INSPECT_PREFIX = "PI_SUBAGENT_INSPECT_JSON:"

export type SubagentStatus = "queued" | "running" | "done" | "failed" | "stopped"

export const isActive = (status: SubagentStatus) => status === "queued" || status === "running"

export interface SubagentRow {
  id: string
  agent: string
  task?: string
  status: SubagentStatus
  /** The tool the child runs now. */
  tool?: string
  tokens?: number
  cost?: number
  startedAt?: number
  endedAt?: number
  /** Elapsed time a foreground child reported, for children without `startedAt`. */
  durationMs?: number
  /** A foreground child's final or latest output. */
  output?: string
  /** Set for background children: what `/subagents-inspect-rpc` takes. */
  inspect?: { asyncId: string; childId?: string }
}

/** `/subagents-inspect-rpc` reply, the parts the roster shows. */
export interface InspectReply {
  requestId: string
  task?: string
  finalOutput?: string
  messages?: { role: string; kind: string; text: string; name?: string; isError?: boolean }[]
  error?: { code: string; message: string }
}

type Json = Record<string, unknown>

const isRecord = (value: unknown): value is Json => typeof value === "object" && value !== null && !Array.isArray(value)
const str = (value: unknown): string | undefined => typeof value === "string" && value ? value : undefined
const num = (value: unknown): number | undefined => typeof value === "number" && Number.isFinite(value) ? value : undefined

function parsePrefixed(lines: readonly string[] | undefined, prefix: string): Json | null {
  const line = lines?.find((candidate) => candidate.startsWith(prefix))
  if (!line) return null
  try {
    const parsed: unknown = JSON.parse(line.slice(prefix.length))
    return isRecord(parsed) ? parsed : null
  } catch {
    return null
  }
}

const ASYNC_STATUS: Record<string, SubagentStatus> = {
  queued: "queued", running: "running", complete: "done", failed: "failed", rejected: "failed", partial: "stopped", paused: "stopped", stopped: "stopped",
}

/** Leaf children of a snapshot node: a run with steps or nested runs stands for them, host monitor steps are no children. */
function asyncLeaves(node: Json, asyncId: string, top: boolean): SubagentRow[] {
  if (node.kind === "host-step") return []
  const children = Array.isArray(node.children) ? node.children.filter(isRecord) : []
  if (children.length > 0) return children.flatMap((child) => asyncLeaves(child, asyncId, false))
  const id = str(node.id)
  if (!id) return []
  const activity = isRecord(node.activity) ? node.activity : {}
  const label = str(node.label)
  return [{
    id: `async:${asyncId}:${top ? "" : id}`,
    agent: str(node.agent) ?? label ?? "subagent",
    task: str(node.progressText),
    status: ASYNC_STATUS[String(node.state)] ?? "running",
    tool: str(activity.currentTool),
    startedAt: num(node.startedAt),
    endedAt: num(node.endedAt),
    inspect: { asyncId, ...(top ? {} : { childId: id }) },
  }]
}

/** The background children of the `subagent-async` widget's lines, in snapshot order. */
export function rowsFromAsyncWidget(lines: readonly string[] | undefined): SubagentRow[] {
  const snapshot = parsePrefixed(lines, ASYNC_PREFIX)
  if (!snapshot || !Array.isArray(snapshot.runs)) return []
  return snapshot.runs.filter(isRecord).flatMap((run) => {
    const asyncId = str(run.id)
    return asyncId ? asyncLeaves(run, asyncId, true) : []
  })
}

const PROGRESS_STATUS: Record<string, SubagentStatus> = { pending: "queued", running: "running", detached: "running", completed: "done", failed: "failed" }

function resultStatus(result: Json): SubagentStatus {
  if (result.stopped === true || result.interrupted === true) return "stopped"
  return result.exitCode === 0 && !str(result.error) ? "done" : "failed"
}

function resultOutput(result: Json): string | undefined {
  return str(result.finalOutput) ?? str(result.error)
}

/** The foreground children of one `subagent` tool call, from its (partial or final) `details`. */
export function rowsFromToolDetails(toolCallId: string, details: unknown, running: boolean): SubagentRow[] {
  if (!isRecord(details) || details.mode === "management") return []
  const results = Array.isArray(details.results) ? details.results.filter(isRecord) : []
  const progress = Array.isArray(details.progress) ? details.progress.filter(isRecord) : []
  const byIndex = new Map(progress.map((entry, position) => [num(entry.index) ?? position, entry]))
  const finished = results.filter((result) => str(result.agent)).map((result, position): SubagentRow => {
    const live = isRecord(result.progress) ? result.progress : byIndex.get(num(result.index) ?? position)
    const usage = isRecord(result.usage) ? result.usage : {}
    const tokens = (num(usage.input) ?? 0) + (num(usage.output) ?? 0)
    return {
      id: `fg:${toolCallId}:${num(result.index) ?? position}`,
      agent: str(result.agent)!,
      task: str(result.task),
      // A result row of a call still running is a child that has not finished yet
      status: running && !live?.status ? "running" : resultStatus(result),
      tokens: tokens || num(live?.tokens),
      cost: num(usage.cost),
      durationMs: num(live?.durationMs),
      output: resultOutput(result),
    }
  })
  const done = new Set(finished.map((row) => row.id))
  const live = progress.filter((entry) => str(entry.agent)).map((entry, position): SubagentRow => {
    const recent = Array.isArray(entry.recentOutput) ? entry.recentOutput.filter((line): line is string => typeof line === "string").join("\n") : undefined
    return {
      id: `fg:${toolCallId}:${num(entry.index) ?? position}`,
      agent: str(entry.agent)!,
      task: str(entry.task),
      status: PROGRESS_STATUS[String(entry.status)] ?? "running",
      tool: str(entry.currentTool),
      tokens: num(entry.tokens),
      durationMs: num(entry.durationMs),
      output: recent || undefined,
    }
  }).filter((row) => !done.has(row.id))
  return [...finished, ...live]
}

/** Every foreground child of this connection's tool runs. */
export function rowsFromToolRuns(runs: Record<string, ToolRun>): SubagentRow[] {
  return Object.entries(runs).flatMap(([toolCallId, run]) => rowsFromToolDetails(toolCallId, run.result?.details, run.running))
}

/** The roster's rows for the current sources: background first, then foreground. */
export function currentRows(runs: Record<string, ToolRun>, widgets: Record<string, string[]>): SubagentRow[] {
  return [...rowsFromAsyncWidget(widgets[ASYNC_WIDGET_KEY]), ...rowsFromToolRuns(runs)]
}

/** Rows shown and the ids the user dismissed (or the next run cleared), which stay out. */
export interface Roster {
  rows: SubagentRow[]
  dismissed: ReadonlySet<string>
}

export const EMPTY_ROSTER: Roster = { rows: [], dismissed: new Set() }

/**
 * Fold the current rows into the roster: known rows update in place, new ones append, dismissed ones stay out.
 * A row that was active and is gone from the sources (its widget retracted) finished, its end is unknown.
 */
export function mergeRoster(roster: Roster, current: readonly SubagentRow[], now = Date.now()): Roster {
  const live = new Map(current.filter((row) => !roster.dismissed.has(row.id)).map((row) => [row.id, row]))
  const rows = roster.rows.map((row) => {
    const next = live.get(row.id)
    live.delete(row.id)
    if (next) return next
    return isActive(row.status) ? { ...row, status: "done" as const, tool: undefined, endedAt: now } : row
  })
  return { ...roster, rows: [...rows, ...live.values()] }
}

export function dismissRows(roster: Roster, ids: readonly string[]): Roster {
  const hidden = new Set(ids)
  return { rows: roster.rows.filter((row) => !hidden.has(row.id)), dismissed: new Set([...roster.dismissed, ...ids]) }
}

/** The finished rows go (dismissed, the user clears them or the next run starts). */
export function dismissFinished(roster: Roster): Roster {
  return dismissRows(roster, roster.rows.filter((row) => !isActive(row.status)).map((row) => row.id))
}

/** Elapsed milliseconds of a row at `now`, if known. */
export function elapsedMs(row: SubagentRow, now: number): number | undefined {
  if (row.startedAt !== undefined) return Math.max(0, (row.endedAt ?? (isActive(row.status) ? now : row.startedAt)) - row.startedAt)
  return row.durationMs
}

export function formatDuration(ms: number): string {
  const seconds = Math.round(ms / 1000)
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  return minutes < 60 ? `${minutes}m ${seconds % 60}s` : `${Math.floor(minutes / 60)}h ${minutes % 60}m`
}

export function formatTokens(count: number): string {
  if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(1)}M`
  return count >= 1_000 ? `${(count / 1_000).toFixed(1)}k` : String(Math.round(count))
}

export function formatCost(cost: number): string {
  return `$${cost < 0.01 ? cost.toFixed(4) : cost.toFixed(2)}`
}

/** The `/subagents-inspect-rpc` command line for `requestId` and a row's target. */
export function inspectCommand(requestId: string, target: NonNullable<SubagentRow["inspect"]>): string {
  return ["/subagents-inspect-rpc", requestId, target.asyncId, target.childId].filter(Boolean).join(" ")
}

/** The inspect reply a `subagent-inspect` widget carries, null for any other lines. */
export function parseInspectReply(lines: readonly string[] | undefined): InspectReply | null {
  const reply = parsePrefixed(lines, INSPECT_PREFIX)
  const requestId = str(reply?.requestId)
  // SAFETY: the reply is versioned JSON from the extension, every InspectReply field but `requestId` (checked here) is optional and read leniently
  return reply && requestId ? reply as unknown as InspectReply : null
}
