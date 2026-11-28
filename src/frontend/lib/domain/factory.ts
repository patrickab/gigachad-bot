// The factory core's JSON shapes (the plugin's `core.ts`) with its field names, as `POST /api/factory/call` answers
// them, and the backend's task preview.

/** Every action of the core, called as `callFactory(action, params)`. */
export type FactoryActionName =
  | "repos.list" | "repos.add"
  | "branches.list"
  | "settings.get" | "settings.set"
  | "models.get" | "models.set"
  | "config.get" | "config.write"
  | "tasks.list" | "tasks.get" | "tasks.wait" | "tasks.timeline" | "tasks.diff" | "tasks.start"
  | "tasks.approve" | "tasks.retry" | "tasks.cancel" | "tasks.remove" | "tasks.message"
  | "waves.start" | "waves.add" | "waves.get" | "waves.list"
  | "waves.retry" | "waves.approve" | "waves.cancel" | "waves.remove"
  | "decisions.list"

/** Codes of the core's refusals, in the `detail.code` of a failed call. */
export type FactoryErrorCode = "invalid" | "not_found" | "busy" | "refused" | "failed" | "factory_plugin_missing"

/** The task actions the core would accept for a task now, as `FactoryTask.actions` lists them. */
export type FactoryTaskAction = "tasks.approve" | "tasks.retry" | "tasks.cancel" | "tasks.remove" | "tasks.message"

/** A JSON schema of one parameter of an action, the keys the core uses. */
export interface FactoryParamSchema {
  type?: string
  description?: string
  enum?: string[]
  minimum?: number
  maximum?: number
  properties?: Record<string, FactoryParamSchema>
}

/** One action of the core's `ACTIONS`: what it does, its parameters, whether it changes anything. */
export interface FactoryActionSpec {
  summary: string
  params: { type: "object"; required: string[]; properties: Record<string, FactoryParamSchema> }
  mutates: boolean
  hostOnly: boolean
}

/** `GET /api/factory/actions`: the core's `ACTIONS`. */
export interface FactoryActions {
  actions: Partial<Record<FactoryActionName, FactoryActionSpec>>
}

/** `repos.add`: the repo it registered (the primary checkout of the path given) and every registered one. */
export interface FactoryRepoAdd {
  repo: string
  repos: string[]
}

export type FactoryMergeMode = "auto" | "review"

/** `models.get` and `models.set`: the user's factory models (`provider/model` ids), the only ones a task may use. */
export interface FactoryModelList {
  models: string[]
}

/**
 * Model ids (`provider/model`), Pi's default where one is missing: per agent of a task, and the project's two sizes
 * (easy tasks of a wave run on `small`, everything else on `big`).
 */
export interface FactoryModels {
  reviewer?: string
  small?: string
  big?: string
}

/** The settings a task runs with (the core's `Settings`): `branch` is the one it starts from and merges into. */
export interface FactorySettings {
  branch: string
  merge: FactoryMergeMode
  maxRounds: number
  review: boolean
  models: FactoryModels
}

/** A repo's settings over the defaults, with `preview`, which only gigachad reads (the core's `ResolvedSettings`). */
export type FactoryResolvedSettings = FactorySettings & { preview: boolean }

/** What a repo's settings file holds: only the keys set (the core's `StoredSettings`). */
export type FactoryStoredSettings = Partial<Omit<FactorySettings, "models">> & { models?: FactoryModels; preview?: boolean; squash?: boolean }

/** `settings.get` and `settings.set`. */
export interface FactoryRepoSettings {
  stored: FactoryStoredSettings
  resolved: FactoryResolvedSettings
  defaults: FactoryResolvedSettings
}

/** One local branch of `branches.list`. */
export interface FactoryBranch {
  name: string
  default: boolean
  hasConfig: boolean
}

/** `config.get`: the repo's `factory.yaml`, or a draft of one with the files it was drawn from. */
export interface FactoryConfig {
  exists: boolean
  text: string | null
  draft: string | null
  source: string[]
}

/** The factory's task states (its `state.ts`); `reviewing`: gates passed, waiting for the wave's reviewer. */
export type FactoryTaskState = "running" | "waiting_approve" | "waiting_answer" | "blocked" | "merging" | "done" | "cancelled" | "reviewing"

/** Tokens and dollars of the coder and the reviews; `input` is uncached input tokens. */
export interface FactoryUsage {
  input: number
  output: number
  cost: number
  cacheRead?: number
  cacheWrite?: number
}

/** A task as every client sees it (the core's `TaskView`). */
export interface FactoryTask {
  id: string
  /** Path of the repo's primary checkout. */
  repo: string
  worktree: string
  /** The task's branch. */
  branch: string
  title: string
  /** Segments joined by `/`: a login or system user alone, or, in a wave, the wave's tag (`plugin/m1/w1`). */
  tag: string
  /** The memory_store of the plan the task belongs to, null without one. */
  plan: string | null
  /** The id of the wave the task belongs to, null outside a wave. */
  wave: string | null
  /** The coder's open question while the state is `waiting_answer`, else null. */
  question: string | null
  createdAt: string
  sessionId: string
  sessionPath: string | null
  /** Null before the task's first `factory-state` entry. */
  state: FactoryTaskState | null
  round: number | null
  reason: string | null
  /** Timestamp of the last `factory-state` entry. */
  at: string | null
  /** A live runner holds the task. */
  running: boolean
  /** The state is `running` or `merging` but no runner lives. */
  interrupted: boolean
  /** The task actions the core would accept now: a client offers exactly these. */
  actions: FactoryTaskAction[]
  settings: FactorySettings | null
  usage: FactoryUsage | null
  /** The commit the task's last merge landed, null before one: deleting the task reverts it. */
  merged: string | null
}

/** Settings over the repo's stored ones for one new task. */
export type FactoryTaskOverrides = Partial<Omit<FactorySettings, "models">> & { models?: FactoryModels }

/** The params of `tasks.start`. */
export interface FactoryTaskStart {
  repo: string
  title: string
  request: string
  overrides?: FactoryTaskOverrides
  /** Omitted: the backend tags the task with the requesting login. */
  tag?: string
}

/** Transitions the core takes with the task id alone (`tasks.<command>`). */
export type FactoryTaskCommand = "approve" | "retry" | "cancel"

/** `tasks.diff`: the task's worktree against where it branched off its target, untracked files as new files. */
export interface FactoryTaskDiff {
  base: string
  diff: string
  files: { path: string; added: number; removed: number }[]
}

/** `tasks.remove`: the removed id and, under `force`, what work it dropped. */
export interface FactoryTaskRemoval {
  removed: string
  dropped: string | null
}

export interface FactoryPreview {
  state: "starting" | "ready" | "failed" | "stopped"
  url: string | null
  error: string | null
}

export type FactoryRoundTrigger = "start" | "hooks" | "gates" | "review" | "protected" | "message" | "retry" | "conflict"

export type FactoryGatePhase = "hooks" | "gates" | "merge_gates" | "after_merge"

export type FactoryReviewStatus = "ok" | "malformed" | "timeout" | "aborted" | "failed" | "no_diff"

export interface FactoryReviewFinding {
  file: string
  line: number
  severity: "blocker" | "major" | "minor"
  text: string
}

export interface FactoryReviewUsage {
  input: number
  output: number
  cacheRead: number
  cacheWrite: number
  cost: number
}

/** `data` of each timeline entry kind: the `factory-*` entry's data unchanged, the coder's and the user's text. */
export interface FactoryTimelineData {
  task: {
    title: string
    request: string
    settings: FactorySettings
    branch: string
    mainBranch: string
    baseSha: string
  }
  round: { round: number; trigger: FactoryRoundTrigger }
  gate: {
    round: number
    phase: FactoryGatePhase
    argv: string[]
    exitCode: number | null
    killed: boolean
    durationMs: number
    outputTail: string
  }
  review: {
    round: number
    status: FactoryReviewStatus
    approved: boolean
    findings: FactoryReviewFinding[]
    model: string
    sessionFile: string
    usage: FactoryReviewUsage
    wallMs: number
  }
  merge: { ok: boolean; commit: string; error: string; depsSynced: boolean }
  /** A task the coder started from a follow-up on this merged task. */
  spawn: { id: string; title: string }
  state: { state: FactoryTaskState; round: number; reason: string }
  coder: { text: string }
  user: { text: string }
  /** The coder asked the user (`factory_ask_supervisor`); the next message answers it. */
  question: { text: string }
}

export type FactoryTimelineKind = keyof FactoryTimelineData

/** One entry of a task's timeline (the core's `TimelineEntry`), discriminated on `kind`. */
export type FactoryTimelineEntry = {
  [Kind in FactoryTimelineKind]: { at: string; kind: Kind; data: FactoryTimelineData[Kind] }
}[FactoryTimelineKind]

/** `tasks.timeline`: the entries after `since`, `next` the cursor for the following read. */
export interface FactoryTimeline {
  entries: FactoryTimelineEntry[]
  next: number
}

/** Where a wave is (the factory's `waves.ts`): waiting for other waves, building its tasks, in review, merging, or resting in one of the others. */
export type FactoryWaveStage = "queued" | "building" | "reviewing" | "rejected" | "merging" | "waiting_approve" | "merged" | "blocked" | "cancelled"

/** One declared task of a wave and, once started, its task (`taskId`) and state ("pending" before the start). */
export interface FactoryWaveTask {
  key: string
  title: string
  after: string[]
  taskId: string | null
  sessionId: string | null
  state: string
  usage: FactoryUsage | null
}

/** One review of a wave's whole diff. */
export interface FactoryWaveReview {
  round: number
  at: string
  sessionId: string | null
  model: string | null
  verdict: "approved" | "rejected" | "failed"
  status: string
  findings: { severity: "blocker" | "major" | "minor"; file: string; line: number | null; text: string }[]
  usage: FactoryUsage | null
  wallMs: number
}

/** A wave as every client sees it (the core's `WaveView`): tasks on one branch, reviewed and merged as a whole. */
export interface FactoryWave {
  id: string
  repo: string
  /** Segments joined by `/`: the prefixes are its track and milestone, the last one the wave (`plugin/m1/w1`). */
  tag: string
  /** The memory_store of the plan the wave belongs to, null without one. */
  plan: string | null
  title: string
  /** The reviewer's instructions. */
  request: string
  branch: string
  target: string
  /** The commit the wave branched from. */
  base: string
  /** The declaring Pi session's id. */
  origin: string | null
  createdAt: string
  /** The ids of the waves it waits for while `queued`. */
  after?: string[]
  stage: FactoryWaveStage
  /** Only when blocked. */
  reason: "conflict" | "gates" | "dirty" | "infra" | "review_failed" | null
  running: boolean
  /** Not resting and no runner. */
  interrupted: boolean
  tasks: FactoryWaveTask[]
  /** Cancelled tasks are in neither. */
  counts: { total: number; done: number }
  reviews: FactoryWaveReview[]
  gates: { at: string; phase: "review" | "merge"; passed: boolean; wallMs: number; log: string | null }[]
  /** The commit range that reached the target. */
  merge: { at: string; from: string; to: string; queueMs: number } | null
  remote: { number: number | null; url: string | null; pushed: string | null; error: string | null } | null
  /** Reviews and tasks, summed. */
  usage: FactoryUsage | null
  /** The `waves.*` actions valid now. */
  actions: string[]
}

const WAVE_STAGE_LABELS: Record<FactoryWaveStage, string> = {
  queued: "queued",
  building: "building",
  reviewing: "reviewing",
  rejected: "rejected",
  merging: "merging",
  waiting_approve: "ready for review",
  merged: "merged",
  blocked: "blocked",
  cancelled: "cancelled",
}

/** A wave's stage as the board and a Pi tab show it: interrupted over the stage, a blocked wave's reason after it. */
export function waveStageLabel(wave: Pick<FactoryWave, "stage" | "reason" | "interrupted">): string {
  if (wave.interrupted) return "interrupted"
  return wave.stage === "blocked" && wave.reason ? `blocked · ${wave.reason}` : WAVE_STAGE_LABELS[wave.stage]
}

/** Whether a wave cannot move without the user: `rejected`, `blocked`, `waiting_approve` or interrupted. */
export function waveNeedsUser(wave: Pick<FactoryWave, "stage" | "interrupted">): boolean {
  return wave.interrupted || wave.stage === "rejected" || wave.stage === "blocked" || wave.stage === "waiting_approve"
}

/**
 * Whether a pi-host `sessions_changed` notice may concern the factory: one of its paths lies under a `/factory/`
 * directory (task and wave records, runner locks, task sessions), or it names none (pi-host could not tell what changed).
 */
export function concernsFactory(event: { type: string; sessionPaths?: unknown }): boolean {
  if (event.type !== "sessions_changed") return false
  const paths = event.sessionPaths
  if (!Array.isArray(paths) || paths.length === 0) return true
  return paths.some((path) => typeof path === "string" && path.includes("/factory/"))
}
