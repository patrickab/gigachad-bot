"use client"

import { useCallback, useState, type ReactNode } from "react"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"
import { Disclosure } from "@/components/ui/Disclosure"
import { ElevatedContainer } from "@/components/ui/ElevatedContainer"
import { useLoadWhile } from "@/hooks/useLoadWhile"
import { readMemory } from "@/lib/backend/api"
import { waveStageLabel, type FactoryTask, type FactoryUsage, type FactoryWave } from "@/lib/domain/factory"
import { cn } from "@/lib/utils"
import { formatSpend } from "./usage"
import { SessionButton, WaveOutcome } from "./WaveOutcome"
import { descriptionPath, descriptionTitle, type WaveGroup } from "./waveTree"

/**
 * The document `plan/<path>.md` of the memory_store `plan`, read once mounted: null until it came back, and for a
 * missing or empty document, a missing memory_store or any failed read.
 */
function useDescription(plan: string | null, path: string): string | null {
  const [text, setText] = useState<string | null>(null)
  const load = useCallback(() => readMemory(plan ?? "", descriptionPath(path)), [plan, path])
  useLoadWhile(plan !== null, load, (read) => setText(read.kind === "document" && read.text.trim() ? read.text : null))
  return plan === null ? null : text
}

/** A description rendered as markdown when its row is opened. */
function DescriptionRow({ text }: { text: string }) {
  return (
    <Disclosure summary="Description" summaryClassName="text-ink-subtle">
      <div className="py-1 text-xs text-ink-muted"><LaTeXMarkdown content={text} compact /></div>
    </Disclosure>
  )
}

/**
 * The description row of the track, milestone or wave at tag `path`: the document `plan/<path>.md` of its plan's
 * memory_store, appearing only once a non-empty document came back.
 */
export function TagGroupDescription({ plan, path }: { plan: string; path: string }) {
  const text = useDescription(plan, path)
  return text === null ? null : <DescriptionRow text={text} />
}

/** A container's header: its progress and title, then its stage and spend on a quiet second line. */
function ContainerHeader({ progress, title, stage, usage }: {
  progress: string
  title: string
  stage?: { label: string; failed: boolean }
  usage: FactoryUsage | null
}) {
  return (
    <span className="flex min-w-0 flex-1 flex-col gap-0.5">
      <span className="flex min-w-0 items-baseline gap-2">
        <span aria-label="Progress" className="shrink-0 text-[11px] tabular-nums text-ink-faint">{progress}</span>
        <span className="min-w-0 truncate text-xs font-semibold text-ink">{title}</span>
      </span>
      {(stage || usage) && (
        <span className="flex min-w-0 items-baseline gap-2 text-[11px]">
          {stage && <span aria-label="Stage" className={cn("min-w-0 truncate", stage.failed ? "text-danger" : "text-ink-faint")}>{stage.label}</span>}
          {usage && <span aria-label="Tokens and cost" className="ml-auto shrink-0 tabular-nums text-ink-faint">{formatSpend(usage)}</span>}
        </span>
      )}
    </span>
  )
}

/** A card-shaped container in a column: a header that opens and closes it, its content below. */
function Container({ label, header, defaultOpen, children }: { label: string; header: ReactNode; defaultOpen: boolean; children: ReactNode }) {
  return (
    <ElevatedContainer role="group" aria-label={label} className="min-w-0 rounded-xl border border-divider/40 shadow-[var(--shadow-md)]">
      <Disclosure summary={header} defaultOpen={defaultOpen} summaryClassName="h-auto items-start rounded-xl px-3 py-2.5 [&>svg]:mt-0.5">
        <div className="flex min-w-0 flex-col gap-2 px-2 pb-2">{children}</div>
      </Disclosure>
    </ElevatedContainer>
  )
}

/** What every container passes down: the done tasks per wave, how to draw one, how to open a Pi session. */
interface ContainerProps {
  /** A wave's done tasks, by wave id, in the order they show. */
  doneTasks: ReadonlyMap<string, readonly FactoryTask[]>
  renderTask: (task: FactoryTask) => ReactNode
  onOpenSession: (sessionId: string) => void
  /** Whether the containers start open: a finished wave's start closed. */
  defaultOpen: boolean
}

/**
 * One wave in its column: `done/total`, title, stage and spend, then while open its done tasks as cards and what
 * happened to it: the declaring supervisor's session, its description, reviews, failed gates, merge and remote.
 */
function WaveContainer({ wave, doneTasks, renderTask, onOpenSession, defaultOpen }: ContainerProps & { wave: FactoryWave }) {
  const done = doneTasks.get(wave.id) ?? []
  const stage = { label: waveStageLabel(wave), failed: wave.interrupted || wave.stage === "blocked" }
  return (
    <Container
      label={wave.title}
      defaultOpen={defaultOpen}
      header={<ContainerHeader progress={`${wave.counts.done}/${wave.counts.total}`} title={wave.title} stage={stage} usage={wave.usage} />}
    >
      {done.map((task) => <div key={task.id}>{renderTask(task)}</div>)}
      <div className="flex flex-col px-1">
        {wave.origin && (
          <div className="group flex items-baseline gap-2 text-[11px] text-ink-subtle">
            <span>supervisor</span>
            <SessionButton label="Open supervisor session" sessionId={wave.origin} onOpenSession={onOpenSession} />
          </div>
        )}
        <WaveOutcome wave={wave} onOpenSession={onOpenSession} />
        {wave.plan != null && <TagGroupDescription plan={wave.plan} path={wave.tag} />}
      </div>
    </Container>
  )
}

/** A track or milestone in a column: `merged/waves`, the first line of its description (or its name), summed spend. */
function GroupContainer({ group, ...props }: ContainerProps & { group: WaveGroup }) {
  const description = useDescription(group.plan, group.path)
  const title = (description && descriptionTitle(description)) || group.name
  return (
    <Container
      label={title}
      defaultOpen={props.defaultOpen}
      header={<ContainerHeader progress={`${group.counts.merged}/${group.counts.total}`} title={title} usage={group.usage} />}
    >
      <WaveContainers group={group} {...props} />
      {description !== null && <div className="px-1"><DescriptionRow text={description} /></div>}
    </Container>
  )
}

/**
 * The waves of one column in their containers: a container per prefix of their tags (track, milestone) wrapping one
 * per wave. A wave with a one-segment tag has no outer container, nor has a wave alone in its group.
 */
export function WaveContainers({ group, ...props }: ContainerProps & { group: WaveGroup }) {
  return (
    <>
      {group.children.map((child) => {
        // A group holding only one wave adds a frame around it and nothing else: draw the wave itself.
        const only = child.children.length === 0 && child.waves.length === 1 ? child.waves[0] : null
        return only ? <WaveContainer key={child.path} wave={only} {...props} /> : <GroupContainer key={child.path} group={child} {...props} />
      })}
      {group.waves.map((wave) => <WaveContainer key={wave.id} wave={wave} {...props} />)}
    </>
  )
}
