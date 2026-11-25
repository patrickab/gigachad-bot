// A task's timeline. Consumes the core's `tasks.timeline` (the task's session as entries, read on from a cursor),
// provides `useTaskTimeline` (the entries, read incrementally) and `groupTimeline` (the entries by round).
import { useEffect, useState } from "react"
import { callFactory } from "@/lib/backend/api"
import type { FactoryRoundTrigger, FactoryTimeline, FactoryTimelineData, FactoryTimelineEntry, FactoryUsage } from "@/lib/domain/factory"
import { errorMessage } from "@/lib/utils"

/** How often the timeline of a task a runner holds is read on: the coder's messages land in its session file. */
export const TIMELINE_POLL_MS = 2000

/** What ran in a round: a hook or gate, the review, the merge, a task the coder started. */
export type TimelineStep = Extract<FactoryTimelineEntry, { kind: "gate" | "review" | "merge" | "spawn" }>

/** A message of the round's conversation: the user's or the coder's text, or the coder's question to the user. */
export type TimelineMessage = Extract<FactoryTimelineEntry, { kind: "coder" | "user" | "question" }>

export interface TimelineRound {
  round: number
  /** What opened the round, null when its `round` entry was not read. */
  trigger: FactoryRoundTrigger | null
  steps: TimelineStep[]
  /** What the user and the coder wrote in the round, the coder's questions too, in order. */
  messages: TimelineMessage[]
  /** The newest state recorded in the round. */
  state: FactoryTimelineData["state"] | null
}

/** A merge request's branches, from its `task` entry. */
export interface TimelineMergeRequest {
  source: string
  target: string
}

export interface Timeline {
  rounds: TimelineRound[]
  /** Every review: tokens and dollars. */
  review: FactoryUsage
  /** Null for a coder's task. */
  mergeRequest: TimelineMergeRequest | null
}

/**
 * Group timeline entries in append order by round.
 *
 * - Gates, reviews and states join the round they name, a merge, a started task, a message and a question (they name
 *   none) the newest round: the extension records a round before the coder turn it opens, and an answer continues it
 * - A round's `state` is the last one written for it: the approval and the merge of a round stay in its group
 */
export function groupTimeline(entries: readonly FactoryTimelineEntry[]): Timeline {
  const rounds: TimelineRound[] = []
  const roundOf = (round: number): TimelineRound => {
    let group = rounds.find((candidate) => candidate.round === round)
    if (!group) {
      group = { round, trigger: null, steps: [], messages: [], state: null }
      rounds.push(group)
    }
    return group
  }
  const review: FactoryUsage = { input: 0, output: 0, cost: 0 }
  let mergeRequest: TimelineMergeRequest | null = null
  for (const entry of entries) {
    switch (entry.kind) {
      case "coder":
      case "user":
      case "question":
        roundOf(rounds.at(-1)?.round ?? 1).messages.push(entry)
        break
      case "round":
        roundOf(entry.data.round).trigger = entry.data.trigger
        break
      case "gate":
        roundOf(entry.data.round).steps.push(entry)
        break
      case "review":
        roundOf(entry.data.round).steps.push(entry)
        review.input += entry.data.usage.input
        review.output += entry.data.usage.output
        review.cost += entry.data.usage.cost
        break
      case "merge":
      case "spawn":
        roundOf(rounds.at(-1)?.round ?? 0).steps.push(entry)
        break
      case "state":
        roundOf(entry.data.round).state = entry.data
        break
      case "task":
        if (entry.data.kind === "merge_request" && entry.data.source) mergeRequest = { source: entry.data.source, target: entry.data.mainBranch }
        break
    }
  }
  return { rounds, review, mergeRequest }
}

interface TimelineReader {
  /** Reads asked for and not yet done: a poll skips while one is. */
  queued: () => number
  /** Read on from the cursor, after the reads asked for before. */
  read: () => void
}

/** Reads of task `taskId`'s timeline one after another, each from the `next` the one before answered. */
function timelineReader(taskId: string, append: (entries: FactoryTimelineEntry[]) => void, report: (error: string) => void): TimelineReader {
  let next = 0
  let queued = 0
  let reads = Promise.resolve()
  return {
    queued: () => queued,
    read: () => {
      queued += 1
      reads = reads.then(async () => {
        try {
          const answer = await callFactory<FactoryTimeline>("tasks.timeline", { id: taskId, since: next })
          next = answer.next
          if (answer.entries.length > 0) append(answer.entries)
          report("")
        } catch (cause) {
          report(errorMessage(cause, "Could not load the timeline"))
        } finally {
          queued -= 1
        }
      })
    },
  }
}

/**
 * The timeline of task `taskId`, in append order.
 *
 * - Reads all of it once, then on from the last answer's `next` whenever `revision` changes (the owner passes a value
 *   that moves with the task) and every `TIMELINE_POLL_MS` while `running` and no read is pending
 * - Reads run one after another, so no entry arrives twice
 * - `taskId` is fixed per mount: key the owner by task
 * - A failed read keeps the entries and sets `error`, the next successful one clears it
 */
export function useTaskTimeline(taskId: string, revision: string, running: boolean): { entries: FactoryTimelineEntry[]; error: string } {
  const [entries, setEntries] = useState<FactoryTimelineEntry[]>([])
  const [error, setError] = useState("")
  const [reader] = useState(() => timelineReader(taskId, (appended) => setEntries((known) => [...known, ...appended]), setError))

  useEffect(() => {
    reader.read()
  }, [reader, revision])

  useEffect(() => {
    if (!running) return
    const timer = setInterval(() => {
      if (reader.queued() === 0) reader.read()
    }, TIMELINE_POLL_MS)
    return () => clearInterval(timer)
  }, [reader, running])

  return { entries, error }
}
