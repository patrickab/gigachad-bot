"use client"

import { useEffect, useState } from "react"
import { Check, ChevronDown, ChevronRight, Circle, LoaderCircle, Square, X, XCircle, type LucideIcon } from "lucide-react"
import { IconButton } from "@/components/ui/IconButton"
import { useNowMs } from "@/hooks/useNowMs"
import { cn } from "@/lib/utils"
import {
  currentRows, dismissFinished, dismissRows, elapsedMs, EMPTY_ROSTER, formatCost, formatDuration, formatTokens, isActive, mergeRoster,
  type InspectReply, type Roster, type SubagentRow, type SubagentStatus,
} from "./subagents"
import type { ToolRun } from "./types"

const STATUS_ICON: Record<SubagentStatus, { icon: LucideIcon; label: string; className: string }> = {
  queued: { icon: Circle, label: "Queued", className: "text-ink-faint" },
  running: { icon: LoaderCircle, label: "Running", className: "animate-spin text-ink-muted" },
  done: { icon: Check, label: "Done", className: "text-ink-subtle" },
  failed: { icon: XCircle, label: "Failed", className: "text-danger" },
  stopped: { icon: Square, label: "Stopped", className: "text-ink-faint" },
}

/**
 * The subagents of a Pi session as a roster state, from this connection's `subagent` tool runs and the `subagent-async` widget.
 *
 * - Finished children stay until dismissed, or until the next run starts (`running` rising), which clears them
 * - A child that vanishes from the sources while active counts as finished
 */
export function useSubagentRoster(runs: Record<string, ToolRun>, widgets: Record<string, string[]>, running: boolean) {
  const [state, setState] = useState<{ roster: Roster; runs: typeof runs; widgets: typeof widgets; running: boolean }>(() => ({ roster: mergeRoster(EMPTY_ROSTER, currentRows(runs, widgets)), runs, widgets, running }))
  let { roster } = state
  // Adjusted while rendering: the sources changed or a run started
  if (state.runs !== runs || state.widgets !== widgets || state.running !== running) {
    if (running && !state.running) roster = dismissFinished(roster)
    roster = mergeRoster(roster, currentRows(runs, widgets))
    setState({ roster, runs, widgets, running })
  }
  const dismiss = (id: string) => setState((current) => ({ ...current, roster: dismissRows(current.roster, [id]) }))
  const clear = () => setState((current) => ({ ...current, roster: dismissFinished(current.roster) }))
  return { rows: roster.rows, dismiss, clear }
}

function Detail({ row, inspect }: { row: SubagentRow; inspect: (row: SubagentRow) => Promise<InspectReply> }) {
  const [reply, setReply] = useState<{ status: SubagentStatus; value: InspectReply } | null>(null)
  const { inspect: target, status } = row
  useEffect(() => {
    if (!target) return
    let cancelled = false
    void inspect(row).then((value) => { if (!cancelled) setReply({ status, value }) })
    return () => { cancelled = true }
  }, [status]) // eslint-disable-line react-hooks/exhaustive-deps -- read when opened and again once the child's state changes
  const value = reply?.value
  const task = value?.task ?? row.task
  const messages = value?.messages ?? []
  const output = value?.finalOutput ?? row.output
  const text = [task && `Task\n${task}`, messages.length > 0 && messages.map((message) => `${message.name ?? message.role}: ${message.text}`).join("\n"), output && `Output\n${output}`].filter(Boolean).join("\n\n")
  return (
    <div className="mb-1 ml-6 max-h-56 overflow-auto whitespace-pre-wrap break-words pr-2 text-ink-muted" aria-label={`${row.agent} output`}>
      {value?.error ? value.error.message : text || (target && !reply ? "Loading" : "No output yet")}
    </div>
  )
}

function Row({ row, now, open, onToggle, onDismiss, inspect }: {
  row: SubagentRow
  now: number
  open: boolean
  onToggle: () => void
  onDismiss: () => void
  inspect: (row: SubagentRow) => Promise<InspectReply>
}) {
  const { icon: Icon, label, className } = STATUS_ICON[row.status]
  const elapsed = elapsedMs(row, now)
  const metrics = [
    row.tokens ? `${formatTokens(row.tokens)} tok` : null,
    row.cost ? formatCost(row.cost) : null,
    elapsed !== undefined ? formatDuration(elapsed) : null,
  ].filter(Boolean).join(" · ")
  return (
    <li className="group/row">
      <div className="flex items-center gap-1">
        <button type="button" aria-expanded={open} onClick={onToggle} className="flex min-w-0 flex-1 items-center gap-2 rounded py-0.5 text-left hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">
          <Icon size={12} aria-label={label} className={cn("shrink-0", className)} />
          <span className="shrink-0 text-ink">{row.agent}</span>
          {row.task && <span className="min-w-0 flex-1 truncate text-ink-muted">{row.task}</span>}
          {!row.task && <span className="flex-1" />}
          {row.tool && <span className="shrink-0 text-ink-faint">{row.tool}</span>}
          {metrics && <span className="shrink-0 tabular-nums text-ink-faint">{metrics}</span>}
        </button>
        <span className="flex h-5 w-5 shrink-0 items-center justify-center">
          {!isActive(row.status) && <button type="button" aria-label={`Dismiss ${row.agent}`} onClick={onDismiss} className="rounded text-ink-faint opacity-0 hover:text-ink focus-visible:opacity-100 focus-visible:outline focus-visible:outline-1 group-hover/row:opacity-100">
            <X size={12} aria-hidden="true" />
          </button>}
        </span>
      </div>
      {open && <Detail row={row} inspect={inspect} />}
    </li>
  )
}

/**
 * Compact roster of a session's subagents pinned above the composer: one row per child (status icon, agent, task, current tool, tokens, cost, elapsed).
 *
 * - Collapsible, shows only its summary line when closed. Renders nothing without children
 * - A row opens its task, transcript window and output: background children read through `inspect`, foreground ones show what their tool call reported
 * - A finished child's row dismisses on hover, the header clears every finished one
 */
export function SubagentRoster({ rows, onDismiss, onClear, inspect }: {
  rows: SubagentRow[]
  onDismiss: (id: string) => void
  onClear: () => void
  inspect: (row: SubagentRow) => Promise<InspectReply>
}) {
  const [collapsed, setCollapsed] = useState(false)
  const [openId, setOpenId] = useState<string | null>(null)
  const running = rows.filter((row) => isActive(row.status)).length
  const now = useNowMs(1000, running > 0)
  if (rows.length === 0) return null
  const finished = rows.length - running
  const summary = [`${rows.length} subagent${rows.length === 1 ? "" : "s"}`, running > 0 ? `${running} running` : null, finished > 0 && running > 0 ? `${finished} done` : null].filter(Boolean).join(" · ")
  const Chevron = collapsed ? ChevronRight : ChevronDown
  return (
    <section aria-label="Subagents" className="mx-auto mb-2 flex w-full max-w-2xl flex-col rounded-xl bg-paper px-3 py-1.5 text-xs">
      <div className="flex items-center justify-between">
        <button type="button" aria-expanded={!collapsed} onClick={() => setCollapsed((value) => !value)} className="flex items-center gap-1 rounded text-ink-muted hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted">
          <Chevron size={12} aria-hidden="true" />
          {summary}
        </button>
        {finished > 0 && <IconButton icon={X} label="Clear finished subagents" onClick={onClear} className="h-5 w-5" />}
      </div>
      {!collapsed && (
        <ul className="mt-1 flex flex-col">
          {rows.map((row) => (
            <Row
              key={row.id}
              row={row}
              now={now}
              open={openId === row.id}
              onToggle={() => setOpenId((current) => current === row.id ? null : row.id)}
              onDismiss={() => onDismiss(row.id)}
              inspect={inspect}
            />
          ))}
        </ul>
      )}
    </section>
  )
}
