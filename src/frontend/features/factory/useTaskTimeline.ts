// A task's pipeline timeline. Consumes `listFactoryEntries` (the factory entries of the task's Pi session and its coder's
// answers), provides `useTaskTimeline` (the entries, read incrementally) and `groupTimeline` (the entries by round).
import { useEffect, useRef, useState } from "react"
import { listFactoryEntries } from "@/lib/backend/api"
import type { FactoryAnswer, FactoryEntry, FactoryEntryData, FactoryRoundTrigger, FactoryTimelineEntry, FactoryUsage } from "@/lib/domain/factory"
import { errorMessage } from "@/lib/utils"

/** What ran in a round: a hook or gate, the review, the merge. */
export type TimelineStep = Extract<FactoryEntry, { customType: "factory-gate" | "factory-review" | "factory-merge" }>

export interface TimelineRound {
  round: number
  /** What opened the round, null when its `factory-round` entry was not read. */
  trigger: FactoryRoundTrigger | null
  steps: TimelineStep[]
  /** What the coder answered in the round's turn, in order. */
  answers: FactoryAnswer[]
  /** The newest state recorded in the round. */
  state: FactoryEntryData["factory-state"] | null
}

export interface Timeline {
  rounds: TimelineRound[]
  /** Every coder answer and every review: tokens (input with cache reads and writes) and dollars. */
  coder: FactoryUsage
  review: FactoryUsage
}

function addUsage(total: FactoryUsage, usage: FactoryUsage): void {
  total.input += usage.input
  total.output += usage.output
  total.cost += usage.cost
}

/**
 * Group factory entries in append order by round.
 *
 * - Gates, reviews and states join the round they name, a merge and a coder answer (they name none) the newest round:
 *   the extension records a round before the coder turn it opens
 * - A round's `state` is the last one written for it: the approval and the merge of a round stay in its group
 */
export function groupTimeline(entries: readonly FactoryTimelineEntry[]): Timeline {
  const rounds: TimelineRound[] = []
  const roundOf = (round: number): TimelineRound => {
    let group = rounds.find((candidate) => candidate.round === round)
    if (!group) {
      group = { round, trigger: null, steps: [], answers: [], state: null }
      rounds.push(group)
    }
    return group
  }
  const coder: FactoryUsage = { input: 0, output: 0, cost: 0 }
  const review: FactoryUsage = { input: 0, output: 0, cost: 0 }
  for (const entry of entries) {
    if (!("customType" in entry)) {
      roundOf(rounds.at(-1)?.round ?? 1).answers.push(entry)
      addUsage(coder, entry.message.usage)
      continue
    }
    switch (entry.customType) {
      case "factory-round":
        roundOf(entry.data.round).trigger = entry.data.trigger
        break
      case "factory-gate":
        roundOf(entry.data.round).steps.push(entry)
        break
      case "factory-review":
        roundOf(entry.data.round).steps.push(entry)
        addUsage(review, { input: entry.data.usage.input + entry.data.usage.cacheRead + entry.data.usage.cacheWrite, output: entry.data.usage.output, cost: entry.data.usage.cost })
        break
      case "factory-merge":
        roundOf(rounds.at(-1)?.round ?? 0).steps.push(entry)
        break
      case "factory-state":
        roundOf(entry.data.round).state = entry.data
        break
      case "factory-task":
        break
    }
  }
  return { rounds, coder, review }
}

/**
 * The factory entries of the Pi session `sessionId` and its coder's answers, in append order.
 *
 * - Reads all of them once, then only those after the last one it has (`since`) whenever `revision` changes: the owner
 *   passes a value that moves when the session file grows
 * - Reads run one after another, so a change during a read is read after it and no entry arrives twice
 * - `sessionId` is fixed per mount once given: key the owner by task
 * - A failed read keeps the entries and sets `error`, the next successful one clears it
 */
export function useTaskTimeline(sessionId: string | undefined, revision: string): { entries: FactoryTimelineEntry[]; error: string } {
  const [entries, setEntries] = useState<FactoryTimelineEntry[]>([])
  const [error, setError] = useState("")
  const cursor = useRef<{ last: string | undefined; reads: Promise<void> }>({ last: undefined, reads: Promise.resolve() })

  useEffect(() => {
    if (!sessionId) return
    const state = cursor.current
    state.reads = state.reads.then(async () => {
      try {
        const appended = await listFactoryEntries(sessionId, state.last)
        if (appended.length > 0) {
          state.last = appended[appended.length - 1].id
          setEntries((known) => [...known, ...appended])
        }
        setError("")
      } catch (cause) {
        setError(errorMessage(cause, "Could not load the timeline"))
      }
    })
  }, [sessionId, revision])

  return { entries, error }
}
