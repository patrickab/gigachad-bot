// Task lists of the factory repos, shared by the board and the task view, and the starts still on their way. Consumes
// `listFactoryTasks`, `createFactoryTask` and the Pi session store, provides `useFactoryTasks`, `useFactoryTask`,
// `useFactoryLaunches`, `launchTask`, `retryLaunch` and `dismissLaunch`.
import { useCallback, useEffect, useMemo, useSyncExternalStore } from "react"
import { usePiSessions } from "@/features/pi/sessionStore"
import { createFactoryTask, listFactoryTasks } from "@/lib/backend/api"
import type { PiSessionInfo } from "@/lib/backend/api/pi"
import type { FactoryTask, FactoryTaskCreate } from "@/lib/domain/factory"
import { errorMessage } from "@/lib/utils"

interface TaskList {
  tasks: FactoryTask[]
  error: string | null
}

const EMPTY: TaskList = { tasks: [], error: null }

/** Last read of each repo, by path. Kept after the last hook unmounts, so a reopened board shows it until the next read. */
const lists = new Map<string, TaskList>()
/** The read of a repo in flight, and the one more read every ask during it shares. */
interface Read {
  current: Promise<void>
  queued: Promise<void> | null
}

const reading = new Map<string, Read>()
const listeners = new Set<() => void>()

function subscribe(listener: () => void): () => void {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

function setList(repo: string, list: TaskList) {
  lists.set(repo, list)
  for (const listener of listeners) listener()
}

async function fetchList(repo: string): Promise<void> {
  try {
    setList(repo, { tasks: await listFactoryTasks(repo), error: null })
  } catch (cause) {
    setList(repo, { tasks: lists.get(repo)?.tasks ?? [], error: errorMessage(cause, "Factory is unreachable") })
  }
}

function startRead(repo: string): Promise<void> {
  const read: Read = { current: fetchList(repo).finally(() => reading.delete(repo)), queued: null }
  reading.set(repo, read)
  return read.current
}

/** Read the tasks of `repo` again. An ask during a read waits for one more read after it, so it never settles on older data. */
function readTasks(repo: string): Promise<void> {
  const read = reading.get(repo)
  if (!read) return startRead(repo)
  read.queued ??= read.current.then(() => startRead(repo))
  return read.queued
}

/** What a task row shows of the Pi sessions: which exist, their factory state and whether their child runs. */
function sessionsSignature(sessions: PiSessionInfo[]): string {
  return sessions
    .map((session) => `${session.sessionId}:${session.factory?.state ?? ""}:${session.factory?.round ?? ""}:${session.factory?.at ?? ""}:${session.live ? Number(session.live.running) : ""}`)
    .join("|")
}

/**
 * Tasks of the repo at path `repo`, newest first.
 *
 * - Read on mount, and again whenever the Pi session store reports a change of a session, its factory state or its
 *   live child (the store follows pi-host's `sessions_changed`). No polling of its own
 * - `refresh` reads again, for a change this page made itself
 * - A failed read keeps the last tasks and sets `error`, the next successful one clears it
 */
export function useFactoryTasks(repo: string): { tasks: FactoryTask[]; error: string | null; refresh: () => Promise<void> } {
  const { sessions } = usePiSessions()
  const signature = useMemo(() => sessionsSignature(sessions), [sessions])
  const list = useSyncExternalStore(subscribe, () => lists.get(repo) ?? EMPTY, () => EMPTY)

  useEffect(() => {
    void readTasks(repo)
  }, [repo, signature])

  const refresh = useCallback(() => readTasks(repo), [repo])
  return { tasks: list.tasks, error: list.error, refresh }
}

function findTask(id: string): FactoryTask | undefined {
  for (const list of lists.values()) {
    const task = list.tasks.find((candidate) => candidate.id === id)
    if (task) return task
  }
  return undefined
}

/** The task `id` as the last read of its repo's list has it, undefined until a `useFactoryTasks` read it. */
export function useFactoryTask(id: string): FactoryTask | undefined {
  return useSyncExternalStore(subscribe, () => findTask(id), () => undefined)
}

/** A start the user sent whose task is not in the list yet: on its way, or refused with `error`. */
export interface FactoryLaunch {
  id: string
  body: FactoryTaskCreate
  /** Why the backend refused the start, null while it runs. */
  error: string | null
}

const NO_LAUNCHES: FactoryLaunch[] = []
/** Starts of each repo, by path, oldest first. Module state like `lists`: a start outlives the board that sent it. */
const launches = new Map<string, FactoryLaunch[]>()
let launchCount = 0

function setLaunches(repo: string, update: (current: FactoryLaunch[]) => FactoryLaunch[]) {
  launches.set(repo, update(launches.get(repo) ?? NO_LAUNCHES))
  for (const listener of listeners) listener()
}

async function runLaunch(launch: FactoryLaunch): Promise<void> {
  const { repo } = launch.body
  try {
    await createFactoryTask(launch.body)
    // The list holds the task before its launch goes, so the card never blinks out.
    await readTasks(repo)
    setLaunches(repo, (current) => current.filter((candidate) => candidate.id !== launch.id))
  } catch (cause) {
    const error = errorMessage(cause, "Could not start the task")
    setLaunches(repo, (current) => current.map((candidate) => candidate.id === launch.id ? { ...candidate, error } : candidate))
  }
}

/**
 * Start a task without waiting for it: the start shows as a launch until the task is in its repo's list, or stays
 * with the backend's refusal until it is retried or dismissed.
 */
export function launchTask(body: FactoryTaskCreate): Promise<void> {
  launchCount += 1
  const launch: FactoryLaunch = { id: `launch-${launchCount}`, body, error: null }
  setLaunches(body.repo, (current) => [...current, launch])
  return runLaunch(launch)
}

/** Send the refused launch `id` of `repo` again. */
export function retryLaunch(repo: string, id: string): Promise<void> {
  const launch = launches.get(repo)?.find((candidate) => candidate.id === id)
  if (!launch) return Promise.resolve()
  const retried = { ...launch, error: null }
  setLaunches(repo, (current) => current.map((candidate) => candidate.id === id ? retried : candidate))
  return runLaunch(retried)
}

export function dismissLaunch(repo: string, id: string) {
  setLaunches(repo, (current) => current.filter((candidate) => candidate.id !== id))
}

/** The launches of the repo at path `repo`, oldest first. */
export function useFactoryLaunches(repo: string): FactoryLaunch[] {
  return useSyncExternalStore(subscribe, () => launches.get(repo) ?? NO_LAUNCHES, () => NO_LAUNCHES)
}
