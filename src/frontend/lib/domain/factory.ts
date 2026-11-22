// The factory core's JSON shapes (the plugin's `core.ts`) with its field names, as `POST /api/factory/call` answers
// them, and the backend's task preview.

/** Every action of the core, called as `callFactory(action, params)`. */
export type FactoryActionName =
  | "repos.list" | "repos.add" | "repos.remove"
  | "branches.list"
  | "settings.get" | "settings.set"
  | "config.get" | "config.write"
  | "tasks.list" | "tasks.get" | "tasks.timeline" | "tasks.diff" | "tasks.start"
  | "tasks.approve" | "tasks.retry" | "tasks.cancel" | "tasks.remove" | "tasks.message"

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

/** Model ids (`provider/model`) of the two agents of a task, Pi's default where one is missing. */
export interface FactoryModels {
  coder?: string
  reviewer?: string
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
export type FactoryStoredSettings = Partial<Omit<FactorySettings, "models">> & { models?: FactoryModels; preview?: boolean }

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

export type FactoryTaskState = "running" | "waiting_approve" | "blocked" | "merging" | "done" | "cancelled"

/** Tokens and dollars of the coder and the reviews; `input` is uncached input tokens. */
export interface FactoryUsage {
  input: number
  output: number
  cost: number
}

/** A task as every client sees it (the core's `TaskView`). */
export interface FactoryTask {
  id: string
  /** Path of the repo's primary checkout. */
  repo: string
  worktree: string
  branch: string
  title: string
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
}

/** Settings over the repo's stored ones for one new task. */
export type FactoryTaskOverrides = Partial<Omit<FactorySettings, "models">> & { models?: FactoryModels }

/** The params of `tasks.start`. */
export interface FactoryTaskStart {
  repo: string
  title: string
  request: string
  overrides?: FactoryTaskOverrides
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
  state: { state: FactoryTaskState; round: number; reason: string }
  coder: { text: string }
  user: { text: string }
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
