// Factory JSON models (snake_case), served unchanged by `/api/factory`, and the factory entries of a task's Pi session.

/** A repo registered with the factory, `path` is its primary checkout and the key every route takes. */
export interface FactoryRepo {
  name: string
  path: string
}

export type FactoryMergeMode = "auto" | "review"

/** Model selectors of the two agents of a task, Pi's default where one is missing. */
export interface FactoryModels {
  coder?: string
  reviewer?: string
}

/** A repo's `factory.yaml`: present, or the draft the backend generated for it. */
export interface FactoryConfig {
  exists: boolean
  draft: string | null
}

/** Settings of a repo, a new task may override any of them. */
export interface FactorySettings {
  merge: FactoryMergeMode
  max_rounds: number
  review: boolean
  preview: boolean
  models: FactoryModels
  /** The branch a task's worktree starts from, is rebased onto and merges into; null for the repo's default branch. */
  branch: string | null
}

export type FactoryTaskState = "running" | "waiting_approve" | "blocked" | "merging" | "done" | "cancelled"

/** The newest `factory-state` entry of a task's session, `at` is its ISO timestamp. */
export interface FactoryState {
  state: FactoryTaskState
  round: number
  reason: string
  at: string
}

/** Live Pi child of a task's session: whether a run streams, waits on a dialog, and how many clients watch. */
export interface FactoryLive {
  running: boolean
  waiting: boolean
  attached: number
}

export interface FactoryTask {
  id: string
  /** Path of the repo's primary checkout. */
  repo: string
  session_id: string
  session_path: string
  /** Worktree path and branch, both null for a task in the current checkout. */
  worktree: string | null
  branch: string | null
  title: string
  created_at: string
  /** Null before the first state, or when pi-host no longer knows the session. */
  factory: FactoryState | null
  /** Null while the session's Pi child is stopped. */
  live: FactoryLive | null
  /** Tokens and dollars of the coder and the reviews, null before a model answered. */
  usage: FactoryUsage | null
}

/** What models used: `input` counts cache reads and writes too, `cost` is in dollars. */
export interface FactoryUsage {
  input: number
  output: number
  cost: number
}

export interface FactoryTaskCreate {
  repo: string
  title: string
  request: string
  overrides: Partial<FactorySettings>
}

/** Task transitions the backend accepts without a body. */
export type FactoryTaskCommand = "approve" | "retry" | "cancel"

/** Unified diff of a task against its base commit, untracked files as new files. */
export interface FactoryTaskDiff {
  base: string
  diff: string
}

export interface FactoryPreview {
  state: "starting" | "ready" | "failed" | "stopped"
  url: string | null
  error: string | null
}

export type FactoryRoundTrigger = "start" | "hooks" | "gates" | "review" | "protected" | "message" | "retry"

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

/** `data` of each factory entry kind, keyed by its `customType`. */
export interface FactoryEntryData {
  "factory-task": {
    title: string
    request: string
    /** The settings in the plugin's shape: `branch` is the one the task starts from and merges into. */
    settings: { merge: FactoryMergeMode; maxRounds: number; review: boolean; models: FactoryModels; branch: string }
    branch: string
    mainBranch: string
    baseSha: string
  }
  "factory-round": { round: number; trigger: FactoryRoundTrigger }
  "factory-gate": {
    round: number
    phase: FactoryGatePhase
    argv: string[]
    exitCode: number | null
    killed: boolean
    durationMs: number
    outputTail: string
  }
  "factory-review": {
    round: number
    status: FactoryReviewStatus
    approved: boolean
    findings: FactoryReviewFinding[]
    model: string
    sessionFile: string
    usage: FactoryReviewUsage
    wallMs: number
  }
  "factory-merge": {
    ok: boolean
    commit: string
    error: string
    depsSynced: boolean
  }
  /** Written right before the merge step that cannot be undone, so a resume can tell whether it landed. */
  "factory-merge-intent": { commit: string; before: string }
  "factory-state": {
    state: FactoryTaskState
    round: number
    reason: string
    /** Current-checkout task, a state that ends a run: the working tree it ended on. */
    tree?: string
    /** Current-checkout task: the HEAD the task builds on. */
    head?: string
    /** Current-checkout task, a run opened on the done task: the diff base from then on. */
    base?: string
  }
}

/** One factory entry of a task's Pi session, discriminated on `customType`. */
export type FactoryEntry = {
  [Kind in keyof FactoryEntryData]: { id: string; timestamp: string; customType: Kind; data: FactoryEntryData[Kind] }
}[keyof FactoryEntryData]

/** An answer of the task's coder, cut to what the timeline shows: its text (no tool calls or thinking), model and usage. */
export interface FactoryAnswer {
  id: string
  timestamp: string
  type: "message"
  message: { role: "assistant"; text: string; model: string | null; stopReason: string | null; usage: FactoryUsage }
}

/** The timeline of a task: its factory entries and the coder's answers, in append order. */
export type FactoryTimelineEntry = FactoryEntry | FactoryAnswer
