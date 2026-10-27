// Factory engine JSON models (snake_case), served unchanged by `/api/factory`.

export type TaskState = "backlog" | "todo" | "in_progress" | "human_review" | "rework" | "blocked" | "merging" | "done" | "cancelled"

export type Phase = "prepare" | "scout" | "plan" | "code" | "hooks" | "gates" | "review" | "preview" | "merge"

export type MergeMode = "auto" | "review"

export interface FactoryOverrides {
  merge?: MergeMode
  max_rounds?: number
  preview?: boolean
  sandbox?: boolean
  models?: Record<string, string>
}

/** Agents a workflow phase runs, in pipeline order. */
export const FACTORY_AGENT_NAMES = ["scout", "planner", "coder", "reviewer"] as const

export type FactoryAgentName = (typeof FACTORY_AGENT_NAMES)[number]

/** Engine-wide settings every repo and workflow may override, `models` maps an agent name to a model selector. */
export interface FactorySettings {
  merge: MergeMode
  max_rounds: number
  sandbox: boolean
  preview: boolean
  models: Record<string, string>
  agent_timeout_s: number
  command_timeout_s: number
}

/** The user's factory defaults: settings plus each agent's full markdown (front matter and prompt). */
export interface FactoryDefaults {
  settings: FactorySettings
  agents: Record<string, string>
}

/** A repo registered with the factory engine, `path` is its main checkout. */
export interface FactoryRepo {
  name: string
  path: string
}

export interface FactoryWorkflow {
  id: string
  name: string
  phases: Phase[]
  merge: MergeMode
  preview: boolean
}

export interface TaskCreate {
  repo: string
  workflow: string
  title: string
  request: string
  overrides: FactoryOverrides
  /** False saves the task to the backlog instead of starting it. */
  start: boolean
}

export interface GateResult {
  argv: string[]
  exit_code: number
  output_tail: string
}

export interface PhaseRun {
  id: number
  phase: Phase
  round: number
  status: "running" | "ok" | "failed"
  started_at: string
  ended_at: string | null
  summary: string
  /** Agent output envelope, shape depends on the phase. */
  envelope: Record<string, unknown> | null
  gate_results: GateResult[]
}

export interface FactoryMessage {
  id: number
  role: "user" | "agent"
  text: string
  phase_run_id: number | null
  created_at: string
}

export interface FactoryPreview {
  status: "starting" | "ready" | "failed" | "stopped"
  url: string
  open_url: string
  error: string | null
}

export interface FactoryTask {
  id: string
  repo: string
  workflow: string
  title: string
  request: string
  state: TaskState
  phase: Phase | null
  round: number
  config: Record<string, unknown>
  worktree: string | null
  branch: string | null
  base_sha: string | null
  error: string | null
  created_at: string
  updated_at: string
}

export interface FactoryTaskDetail extends FactoryTask {
  phases: PhaseRun[]
  messages: FactoryMessage[]
  preview: FactoryPreview | null
}

/** Task transitions the engine accepts without a body. */
export type TaskAction = "start" | "approve" | "cancel" | "retry"

export interface FactoryEvent {
  seq: number
  task_id: string
  kind: string
  data: Record<string, unknown>
  created_at: string
}
