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

/** Settings of a repo, a new task may override any of them. */
export interface FactorySettings {
  merge: FactoryMergeMode
  max_rounds: number
  review: boolean
  /** True runs a task in its own worktree and branch, false in the repo's current checkout. */
  worktree: boolean
  preview: boolean
  models: FactoryModels
  /** Source and target branch of a worktree task, null for the repo's `main`, else `master`. */
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
    settings: FactorySettings
    worktree: boolean
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
    kind: "merge" | "commit"
    commit: string
    error: string
    depsSynced: boolean
    /** Current-checkout commit: the working tree right after it. */
    tree?: string | null
    /** Current-checkout commit: committed paths the user had also changed, uncommitted, before the task. */
    mixed?: string[]
  }
  /** Written right before the merge step that cannot be undone, so a resume can tell whether it landed. */
  "factory-merge-intent": { kind: "merge" | "commit"; commit: string; before: string }
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
