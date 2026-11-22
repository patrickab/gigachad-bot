// Task lists of the factory repos, shared by the board and the task view, and the starts still on their way. Consumes
// the core's `tasks.list` and `tasks.start` and pi-host's `sessions_changed` notice, provides `useFactoryTasks`,
// `useFactoryTask`, `useFactoryLaunches`, `launchTask`, `retryLaunch` and `dismissLaunch`.
import { useCallback, useEffect, useSyncExternalStore } from "react"
import { observePiSocket, type HostEvent } from "@/features/pi/piSocket"
import { callFactory } from "@/lib/backend/api"
import type { FactoryTask, FactoryTaskStart } from "@/lib/domain/factory"
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
/** Mounted `useFactoryTasks` hooks per repo path: the repos `sessions_changed` reads again. */
const watchers = new Map<string, number>()
let stopLive: (() => void) | null = null

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
    const { tasks } = await callFactory<{ tasks: FactoryTask[] }>("tasks.list", { repo })
    setList(repo, { tasks: [...tasks].sort((left, right) => right.createdAt.localeCompare(left.createdAt)), error: null })
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

function readWatched() {
  for (const repo of watchers.keys()) void readTasks(repo)
}

/**
 * Whether a `sessions_changed` notice may concern a task: one of its paths lies under a `/factory/` directory (task
 * records, runner locks, task sessions), or it names none (pi-host could not tell what changed).
 */
export function concernsFactory(event: HostEvent): boolean {
  if (event.type !== "sessions_changed") return false
  const paths = event.sessionPaths
  if (!Array.isArray(paths) || paths.length === 0) return true
  return paths.some((path) => typeof path === "string" && path.includes("/factory/"))
}

/** pi-host announces a change of a task (its record, lock or session file) as `sessions_changed`, a reopened socket may have missed one. */
function startLive(): () => void {
  return observePiSocket({
    onHostEvent: (event: HostEvent) => {
      if (concernsFactory(event)) readWatched()
    },
    onOpen: readWatched,
  }, { hold: true })
}

function watchRepo(repo: string): () => void {
  watchers.set(repo, (watchers.get(repo) ?? 0) + 1)
  stopLive ??= startLive()
  return () => {
    const left = (watchers.get(repo) ?? 1) - 1
    if (left > 0) watchers.set(repo, left)
    else watchers.delete(repo)
    if (watchers.size > 0) return
    stopLive?.()
    stopLive = null
  }
}

/**
 * Tasks of the repo at path `repo`, newest first, every task the core holds for it (started here or from the CLI).
 *
 * - Read on mount, and again on pi-host's `sessions_changed` notice of a factory path and on a reopened socket. No
 *   polling
 * - `refresh` reads again, for a change this page made itself
 * - A failed read keeps the last tasks and sets `error`, the next successful one clears it
 */
export function useFactoryTasks(repo: string): { tasks: FactoryTask[]; error: string | null; refresh: () => Promise<void> } {
  const list = useSyncExternalStore(subscribe, () => lists.get(repo) ?? EMPTY, () => EMPTY)

  useEffect(() => {
    const unwatch = watchRepo(repo)
    void readTasks(repo)
    return unwatch
  }, [repo])

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
  body: FactoryTaskStart
  /** Why the core refused the start, null while it runs. */
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
    await callFactory<FactoryTask>("tasks.start", { ...launch.body })
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
 * with the core's refusal until it is retried or dismissed.
 */
export function launchTask(body: FactoryTaskStart): Promise<void> {
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
