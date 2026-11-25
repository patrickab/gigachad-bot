"use client"

import { useCallback, useState, type ReactNode } from "react"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"
import { Disclosure } from "@/components/ui/Disclosure"
import { useLoadWhile } from "@/hooks/useLoadWhile"
import { readMemory } from "@/lib/backend/api"
import type { FactoryTask, FactoryUsage, FactoryWave, FactoryWaveStage } from "@/lib/domain/factory"
import { cn } from "@/lib/utils"
import { STATE_ORDER, shownState } from "./states"
import { formatSpend } from "./usage"
import { descriptionPath, waveName, type WaveGroup } from "./waveTree"

const STAGE_LABELS: Record<FactoryWaveStage, string> = {
  building: "building",
  reviewing: "reviewing",
  rejected: "rejected",
  merging: "merging",
  waiting_approve: "ready for review",
  merged: "merged",
  blocked: "blocked",
  cancelled: "cancelled",
}

/** A wave's stage as its row shows it: interrupted over the stage, a blocked wave's reason after it. */
export function waveStageLabel(wave: Pick<FactoryWave, "stage" | "reason" | "interrupted">): string {
  if (wave.interrupted) return "interrupted"
  return wave.stage === "blocked" && wave.reason ? `blocked · ${wave.reason}` : STAGE_LABELS[wave.stage]
}

/**
 * The description row of the track, milestone or wave at tag `path`: the document `plan/<path>.md` of its plan's
 * memory_store rendered as markdown when opened. Read as soon as it mounts (when its row opens), it appears only once a
 * non-empty document came back: a missing document or memory_store (or any failed read) renders nothing.
 */
export function TagGroupDescription({ plan, path }: { plan: string; path: string }) {
  const [text, setText] = useState<string | null>(null)
  const load = useCallback(() => readMemory(plan, descriptionPath(path)), [plan, path])
  useLoadWhile(true, load, (read) => setText(read.kind === "document" && read.text.trim() ? read.text : null))
  if (text === null) return null
  return (
    <Disclosure summary="Description" summaryClassName="text-ink-subtle">
      <div className="py-1 text-xs text-ink-muted"><LaTeXMarkdown content={text} compact /></div>
    </Disclosure>
  )
}

/** A row's name, its `done/total` and its spend. */
function RowSummary({ name, title, counts, usage, stage }: {
  name: string
  title?: string
  counts: { total: number; done: number }
  usage: FactoryUsage | null
  stage?: { label: string; failed: boolean }
}) {
  return (
    <span className="flex min-w-0 items-baseline gap-2">
      <span className="shrink-0">{name}</span>
      {title && <span className="min-w-0 truncate text-ink-subtle">{title}</span>}
      <span aria-label="Progress" className="shrink-0 text-[11px] tabular-nums text-ink-faint">{counts.done}/{counts.total}</span>
      {stage && <span aria-label="Stage" className={cn("shrink-0 text-[11px]", stage.failed ? "text-danger" : "text-ink-faint")}>{stage.label}</span>}
      {usage && <span aria-label="Tokens and cost" className="shrink-0 text-[11px] tabular-nums text-ink-faint">{formatSpend(usage)}</span>}
    </span>
  )
}

/** Board order of a wave's tasks: by shown state, then newest first. */
function stateOrder(left: FactoryTask, right: FactoryTask): number {
  return STATE_ORDER.indexOf(shownState(left)) - STATE_ORDER.indexOf(shownState(right)) || right.createdAt.localeCompare(left.createdAt)
}

/** One wave: its row, and while open its description, its tasks as cards and its specs not started yet as quiet rows. */
function WaveNode({ wave, tasks, renderTask }: { wave: FactoryWave; tasks: readonly FactoryTask[]; renderTask: (task: FactoryTask) => ReactNode }) {
  const own = tasks.filter((task) => task.wave === wave.id).sort(stateOrder)
  const started = new Set(own.map((task) => task.id))
  const pending = wave.tasks.filter((spec) => spec.taskId === null || !started.has(spec.taskId))
  const stage = { label: waveStageLabel(wave), failed: wave.interrupted || wave.stage === "blocked" }
  return (
    <Disclosure summary={<RowSummary name={waveName(wave)} title={wave.title} counts={wave.counts} usage={wave.usage} stage={stage} />}>
      <div className="flex flex-col pl-5">
        {wave.plan != null && <TagGroupDescription plan={wave.plan} path={wave.tag} />}
        {own.length > 0 && (
          <div className="grid grid-cols-[repeat(auto-fill,minmax(15rem,1fr))] gap-2.5 py-2">
            {own.map((task) => <div key={task.id}>{renderTask(task)}</div>)}
          </div>
        )}
        {pending.length > 0 && (
          <ul aria-label="Pending" className="flex flex-col gap-1 py-1">
            {pending.map((spec) => <li key={spec.key} className="truncate text-xs text-ink-faint">{spec.title}</li>)}
          </ul>
        )}
      </div>
    </Disclosure>
  )
}

function GroupNode({ group, tasks, renderTask }: { group: WaveGroup; tasks: readonly FactoryTask[]; renderTask: (task: FactoryTask) => ReactNode }) {
  return (
    <Disclosure summary={<RowSummary name={group.name} counts={group.counts} usage={group.usage} />}>
      <div className="flex flex-col pl-5">
        {group.plan != null && <TagGroupDescription plan={group.plan} path={group.path} />}
        <GroupContent group={group} tasks={tasks} renderTask={renderTask} />
      </div>
    </Disclosure>
  )
}

function GroupContent({ group, tasks, renderTask }: { group: WaveGroup; tasks: readonly FactoryTask[]; renderTask: (task: FactoryTask) => ReactNode }) {
  return (
    <>
      {group.children.map((child) => <GroupNode key={child.path} group={child} tasks={tasks} renderTask={renderTask} />)}
      {group.waves.map((wave) => <WaveNode key={wave.id} wave={wave} tasks={tasks} renderTask={renderTask} />)}
    </>
  )
}

/**
 * The waves above the board's columns: a row per track and milestone (the prefixes of the waves' tags, with the sums
 * of their waves) and per wave (name, title, `done/total`, stage, spend), closed until opened. `renderTask` draws a
 * wave's task as the columns do.
 */
export function WaveTree({ root, tasks, renderTask }: { root: WaveGroup; tasks: readonly FactoryTask[]; renderTask: (task: FactoryTask) => ReactNode }) {
  return (
    <section aria-label="Waves" className="flex flex-col">
      <GroupContent group={root} tasks={tasks} renderTask={renderTask} />
    </section>
  )
}
