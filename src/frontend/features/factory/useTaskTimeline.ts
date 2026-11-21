// A task's pipeline timeline. Consumes `listFactoryEntries` (the factory entries of the task's Pi session), provides
// `useTaskTimeline` (the entries, read incrementally) and `groupTimeline` (the entries by round).
import { useEffect, useRef, useState } from "react"
import { listFactoryEntries } from "@/lib/backend/api"
import type { FactoryEntry, FactoryEntryData, FactoryRoundTrigger } from "@/lib/domain/factory"
import { errorMessage } from "@/lib/utils"

/** What ran in a round: a hook or gate, the review, the merge. */
export type TimelineStep = Extract<FactoryEntry, { customType: "factory-gate" | "factory-review" | "factory-merge" }>

export interface TimelineRound {
  round: number
  /** What opened the round, null when its `factory-round` entry was not read. */
  trigger: FactoryRoundTrigger | null
  steps: TimelineStep[]
  /** The newest state recorded in the round. */
  state: FactoryEntryData["factory-state"] | null
}

export interface Timeline {
  rounds: TimelineRound[]
  /** Cost of every review, in dollars. */
  reviewCost: number
}

/**
 * Group factory entries in append order by round.
 *
 * - Gates, reviews and states join the round they name, a merge (it names none) the newest round
 * - A round's `state` is the last one written for it: the approval and the merge of a round stay in its group
 */
export function groupTimeline(entries: readonly FactoryEntry[]): Timeline {
  const rounds: TimelineRound[] = []
  const roundOf = (round: number): TimelineRound => {
    let group = rounds.find((candidate) => candidate.round === round)
    if (!group) {
      group = { round, trigger: null, steps: [], state: null }
      rounds.push(group)
    }
    return group
  }
  let reviewCost = 0
  for (const entry of entries) {
    switch (entry.customType) {
      case "factory-round":
        roundOf(entry.data.round).trigger = entry.data.trigger
        break
      case "factory-gate":
        roundOf(entry.data.round).steps.push(entry)
        break
      case "factory-review":
        roundOf(entry.data.round).steps.push(entry)
        reviewCost += entry.data.usage.cost
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
  return { rounds, reviewCost }
}

/**
 * The factory entries of the Pi session `sessionId`, in append order.
 *
 * - Reads all of them once, then only those after the last one it has (`since`) whenever `revision` changes: the owner
 *   passes a value that moves when the session file grows
 * - Reads run one after another, so a change during a read is read after it and no entry arrives twice
 * - `sessionId` is fixed per mount once given: key the owner by task
 * - A failed read keeps the entries and sets `error`, the next successful one clears it
 */
export function useTaskTimeline(sessionId: string | undefined, revision: string): { entries: FactoryEntry[]; error: string } {
  const [entries, setEntries] = useState<FactoryEntry[]>([])
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
