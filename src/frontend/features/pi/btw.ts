/**
 * The `/btw` side questions of the user's Pi extension `btw` (`~/.pi/agent/extensions/btw/`, README "Contract with gigachad"):
 * only the wire shapes and the parsing of its `btw_state` widget live here, all btw logic stays in the extension.
 */

export type BtwStatus = "streaming" | "done" | "error" | "cancelled"

export interface BtwFollowUp { question: string; answer: string; status: BtwStatus; error?: string; createdAt: string }

export interface BtwThread {
  id: string
  question: string
  answer: string
  status: BtwStatus
  error?: string
  followUps: BtwFollowUp[]
  createdAt: string
}

export const BTW_WIDGET_KEY = "btw_state"
const STATUSES = new Set<string>(["streaming", "done", "error", "cancelled"])
/** The extension's commands: sent at once, also mid-run, never held, steered or queued. */
const BTW_COMMAND = /^\/btw(?:-(?:followup|cancel|dismiss|list|action))?(?:\s|$)/

export function isBtwCommand(text: string): boolean {
  return BTW_COMMAND.test(text)
}

function isTurn(value: unknown): value is { question: string; answer: string; status: BtwStatus; error?: string } {
  const turn = value as Record<string, unknown> | null
  return typeof turn === "object" && turn !== null && typeof turn.question === "string" && typeof turn.answer === "string"
    && typeof turn.status === "string" && STATUSES.has(turn.status)
}

function isThread(value: unknown): value is BtwThread {
  const thread = value as Record<string, unknown> | null
  return isTurn(value) && typeof (thread as Record<string, unknown>).id === "string"
    && Array.isArray(thread?.followUps) && thread.followUps.every(isTurn)
}

/**
 * The threads a `setWidget` record for `btw_state` carries: none without `widgetLines` (widget cleared), null for an
 * unknown version or a malformed line (keep what is shown).
 */
export function parseBtwWidget(record: { widgetLines?: unknown; [field: string]: unknown }): BtwThread[] | null {
  const lines = record.widgetLines
  if (!Array.isArray(lines) || lines.length === 0) return []
  try {
    const event = JSON.parse(String(lines[0])) as { type?: unknown; version?: unknown; state?: { threads?: unknown } }
    if (event.type !== "btw_state" || event.version !== 1 || !Array.isArray(event.state?.threads)) return null
    return event.state.threads.filter(isThread)
  } catch {
    return null
  }
}
