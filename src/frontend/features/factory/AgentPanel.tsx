"use client"

import { useEffect, useRef, useState } from "react"
import { Send } from "lucide-react"
import { InlineConfirm } from "@/components/ui/InlineConfirm"
import { cn, errorMessage } from "@/lib/utils"
import { ErrorText } from "@/components/ui/ErrorText"
import { runFactoryTaskAction, sendFactoryMessage } from "@/lib/backend/api"
import { PHASE_LABELS, STATE_LABELS, allowedActions, type StatusTone } from "./states"
import { StatusDot } from "./StatusDot"
import { FIELD_CLASS, ICON_BUTTON_CLASS, PRIMARY_BUTTON_CLASS, QUIET_BUTTON_CLASS } from "./styles"
import type { FactoryTaskDetail, GateResult, PhaseRun, TaskAction } from "@/lib/domain/types"

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((entry): entry is string => typeof entry === "string") : []
}

function Gate({ gate }: { gate: GateResult }) {
  const passed = gate.exit_code === 0
  return (
    <li className="text-[11px]">
      <div className="flex items-baseline gap-2">
        <code className="min-w-0 flex-1 truncate font-mono text-ink-muted">{gate.argv.join(" ")}</code>
        <span className={cn("shrink-0", passed ? "text-diff-added" : "text-diff-removed")}>{passed ? "passed" : `exit ${gate.exit_code}`}</span>
      </div>
      {!passed && gate.output_tail && (
        <details className="mt-1">
          <summary className="cursor-pointer text-ink-subtle hover:text-ink">Output</summary>
          <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap rounded bg-surface p-2 font-mono text-[10px] text-ink-muted">{gate.output_tail}</pre>
        </details>
      )}
    </li>
  )
}

function ListBlock({ label, items }: { label: string; items: string[] }) {
  if (items.length === 0) return null
  return (
    <div className="mt-1">
      <p className="text-[11px] text-ink-subtle">{label}</p>
      <ul className="mt-0.5 flex list-disc flex-col gap-0.5 pl-4 text-[11px] text-ink-muted">
        {items.map((item, index) => <li key={index} className="whitespace-pre-wrap">{item}</li>)}
      </ul>
    </div>
  )
}

const RUN_TONE: Record<PhaseRun["status"], StatusTone> = {
  running: "active",
  ok: "done",
  failed: "error",
}

const SECTION_TITLE_CLASS = "pb-2 text-[11px] font-bold uppercase tracking-[0.2em] text-ink-muted"

/** One phase run on the timeline: a dot on the rail, label, summary and its details. */
function PhaseEntry({ run }: { run: PhaseRun }) {
  const question = typeof run.envelope?.question === "string" ? run.envelope.question : ""
  return (
    <li className="relative pb-3 pl-5 last:pb-0">
      <span aria-hidden className="absolute bottom-0 left-[2.5px] top-3 w-px bg-divider/50" />
      <span className="absolute left-0 top-[5px]"><StatusDot tone={RUN_TONE[run.status]} /></span>
      <div className="flex items-baseline gap-2 text-xs">
        <span className="min-w-0 flex-1 truncate font-medium text-ink">{PHASE_LABELS[run.phase]}{run.round > 0 ? ` · round ${run.round}` : ""}</span>
        <span className={cn("shrink-0 text-[11px]", run.status === "failed" ? "text-danger" : "text-ink-subtle")}>{run.status}</span>
      </div>
      {run.summary && <p className="mt-0.5 whitespace-pre-wrap text-[11px] leading-relaxed text-ink-muted">{run.summary}</p>}
      <ListBlock label="Plan" items={run.phase === "plan" ? stringList(run.envelope?.steps) : []} />
      <ListBlock label="Findings" items={run.phase === "review" ? stringList(run.envelope?.findings) : []} />
      {question && <p className="mt-1 whitespace-pre-wrap text-[11px] text-ink">Question: {question}</p>}
      {run.gate_results.length > 0 && <ul className="mt-1 flex flex-col gap-1">{run.gate_results.map((gate, index) => <Gate key={index} gate={gate} />)}</ul>}
    </li>
  )
}

function Composer({ taskId, disabled, onSent, onFailed }: { taskId: string; disabled: boolean; onSent: () => void; onFailed: (message: string) => void }) {
  const [text, setText] = useState("")
  const [sending, setSending] = useState(false)

  const send = async () => {
    const trimmed = text.trim()
    if (!trimmed || sending || disabled) return
    setSending(true)
    try {
      await sendFactoryMessage(taskId, trimmed)
      setText("")
      onSent()
    } catch (cause) {
      onFailed(errorMessage(cause, "Could not send the message"))
    } finally {
      setSending(false)
    }
  }

  return (
    <div className="flex shrink-0 items-end gap-1 border-t border-divider/50 p-2">
      <textarea
        aria-label="Message to the agent"
        placeholder="Message the agent"
        value={text}
        rows={2}
        onChange={(event) => setText(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
            event.preventDefault()
            void send()
          }
        }}
        className={cn(FIELD_CLASS, "min-w-0 flex-1 resize-none py-1.5 text-xs")}
      />
      <button type="button" aria-label="Send message" disabled={disabled || sending || text.trim() === ""} onClick={() => void send()} className={ICON_BUTTON_CLASS}>
        <Send className="h-4 w-4" />
      </button>
    </div>
  )
}

const PRIMARY_ACTIONS = [
  { action: "start", label: "Start" },
  { action: "approve", label: "Approve" },
  { action: "retry", label: "Retry" },
] as const

/** Side panel of a task: state actions, request, phase runs with gate results and review findings, messages and the composer. */
export function AgentPanel({ task, onChanged }: { task: FactoryTaskDetail; onChanged: () => void }) {
  const [busy, setBusy] = useState(false)
  const [confirmingCancel, setConfirmingCancel] = useState(false)
  const [error, setError] = useState("")
  const scroller = useRef<HTMLDivElement>(null)
  const allowed = allowedActions(task.state)

  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight })
  }, [task.messages.length])

  const run = async (action: TaskAction) => {
    setBusy(true)
    setError("")
    try {
      await runFactoryTaskAction(task.id, action)
      onChanged()
    } catch (cause) {
      setError(errorMessage(cause, `Could not ${action} the task`))
    } finally {
      setBusy(false)
      setConfirmingCancel(false)
    }
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex h-12 shrink-0 items-center gap-1 border-b border-divider/20 px-4">
        <span className="min-w-0 flex-1 truncate text-[11px] font-bold uppercase tracking-[0.2em] text-ink-muted">{STATE_LABELS[task.state]}</span>
        {PRIMARY_ACTIONS.map(({ action, label }) => allowed[action] && <button key={action} type="button" disabled={busy} onClick={() => void run(action)} className={PRIMARY_BUTTON_CLASS}>{label}</button>)}
        {allowed.cancel && !confirmingCancel && <button type="button" disabled={busy} onClick={() => setConfirmingCancel(true)} className={QUIET_BUTTON_CLASS}>Cancel</button>}
      </div>
      {confirmingCancel && <div className="px-4 pt-2"><InlineConfirm question="Cancel this task?" busy={busy} onYes={() => void run("cancel")} onNo={() => setConfirmingCancel(false)} /></div>}
      <ErrorText className="px-4 pt-2 text-[11px]">{error || task.error}</ErrorText>
      <div ref={scroller} className="min-h-0 flex-1 space-y-6 overflow-y-auto px-4 py-4 scrollbar-thin">
        <section aria-label="Request">
          <h3 className={SECTION_TITLE_CLASS}>Request</h3>
          <p className="whitespace-pre-wrap rounded-xl border border-divider/40 bg-surface-elevated/60 px-3 py-2.5 text-xs leading-relaxed text-ink">{task.request}</p>
        </section>
        {task.phases.length > 0 && (
          <section aria-label="Phases">
            <h3 className={SECTION_TITLE_CLASS}>Phases</h3>
            <ol>{task.phases.map((phaseRun) => <PhaseEntry key={phaseRun.id} run={phaseRun} />)}</ol>
          </section>
        )}
        {task.messages.length > 0 && (
          <section aria-label="Messages">
            <h3 className={SECTION_TITLE_CLASS}>Messages</h3>
            <ul className="flex flex-col gap-2">
              {task.messages.map((message) => (
                <li
                  key={message.id}
                  className={cn(
                    "max-w-[85%] whitespace-pre-wrap rounded-2xl px-3 py-2 text-xs leading-relaxed",
                    message.role === "user" ? "self-end rounded-br-md bg-surface-elevated text-ink" : "self-start rounded-bl-md border border-divider/40 text-ink-muted",
                  )}
                >
                  {message.text}
                </li>
              ))}
            </ul>
          </section>
        )}
      </div>
      {allowed.message && <Composer taskId={task.id} disabled={busy} onSent={onChanged} onFailed={setError} />}
    </div>
  )
}
