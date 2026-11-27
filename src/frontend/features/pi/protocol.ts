/** The pi-host socket protocol as the browser reads it: the frames of one session and the Pi RPC answers it uses. */
import type { ResyncEntries, SessionEntries } from "./transcript"
import type { PiCommand, PiModel, PiState } from "./types"

/** One Pi stdout record; records the browser does not read stay inert. */
export type PiRecord = { type: string; id?: string; [field: string]: unknown }

/** pi-host's view of the session's open UI when a client attaches (M3). */
export interface AttachedUi {
  dialogs: PiRecord[]
  statuses: Record<string, string>
  /** Open text widgets by key; absent from an older pi-host. */
  widgets?: Record<string, string[]>
  queue?: { steering: string[]; followUp: string[] }
}

/** A frame pi-host sends about one session (`pi-host/src/supervisor/session.ts`). */
export type HostFrame = { sessionId: string; streamId?: string | null; sequence?: number } & (
  | { type: "attached"; state: PiState; entries: SessionEntries; ui: AttachedUi }
  /** `state` and `ui` like `attached`, tolerated missing. */
  | ({ type: "resync"; state?: PiState; ui?: AttachedUi } & ResyncEntries)
  | { type: "pi"; record: PiRecord }
  | { type: "ui_closed"; id: string }
  | { type: "replay_start" | "replay_end" | "lock_held" }
  | { type: "exit"; code: number | null; stderrTail?: string }
  | { type: "error"; message: string }
)

/** A frame the browser sends about one session. */
export type ClientFrame = { sessionId: string } & ({ type: "attach" | "detach" } | { type: "resync"; cursor: string | null } | { type: "command"; command: Record<string, unknown> })

/** `get_session_stats` data the top bar reads: the session's cost and the current context window use (F3). */
export interface PiSessionStats {
  cost: number
  /** Cost of the session's subagents and factory tasks, added by the view: Pi's own stats cover the main agent only. */
  subCost?: number
  /** Subagents and factory tasks of the session working now, added by the view. */
  subActive?: number
  /** Tokens over the whole session; `input` is uncached input. */
  tokens?: { input: number; output: number; cacheRead: number; cacheWrite: number }
  /** Absent without a model, `tokens`/`percent` null right after a compaction until the next reply. */
  contextUsage?: { tokens: number | null; contextWindow: number; percent: number | null }
}

/** The `data` of a successful answer, per command whose data the browser reads. */
interface PiCommandData {
  get_state: PiState
  get_commands: { commands: PiCommand[] }
  get_available_models: { models: PiModel[] }
  get_available_thinking_levels: { levels: string[] }
  get_session_stats: PiSessionStats
  clear_queue: { steering: string[]; followUp: string[] }
}

/** Pi's answer to the RPC command `C`. */
export type PiResponse<C extends string = string> =
  | { success: true; data: C extends keyof PiCommandData ? PiCommandData[C] : unknown }
  | { success: false; data?: undefined; error?: string }
