"use client"

import { useEffect, useRef, useState } from "react"
import { ListChecks } from "lucide-react"
import { CopyAction, TurnCard, TurnRegion, TurnSurface } from "@gigachad/chat-ui"
import "@gigachad/chat-ui/turn.css"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"
import { COMPOSER_ICON_BUTTON, Composer } from "@/components/ui/Composer"
import { ErrorText } from "@/components/ui/ErrorText"
import { InlineConfirm } from "@/components/ui/InlineConfirm"
import { StatusDot, type StatusTone } from "@/components/ui/StatusDot"
import { FilePanel } from "@/features/editor/FilePanel"
import { useAsyncAction } from "@/hooks/useAsyncAction"
import { runFactoryTaskAction, sendFactoryMessage } from "@/lib/backend/api"
import { cn, errorMessage } from "@/lib/utils"
import { AttachImagesButton, StagedImages, toFactoryImages, useImageAttachments } from "./ImageAttachments"
import { PHASE_LABELS, STATE_LABELS, allowedActions } from "./states"
import { PRIMARY_BUTTON_CLASS, QUIET_BUTTON_CLASS } from "./styles"
import type { FactoryMessage, FactoryTaskDetail, GateResult, PhaseRun, TaskAction } from "@/lib/domain/factory"

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

/** One phase run on the timeline: a dot on the rail, label, summary and its details. `hideSummary` when a reply already says it. */
function PhaseEntry({ run, hideSummary }: { run: PhaseRun; hideSummary: boolean }) {
  const question = typeof run.envelope?.question === "string" ? run.envelope.question : ""
  return (
    <li className="relative pb-3 pl-5 last:pb-0">
      <span aria-hidden className="absolute bottom-0 left-[2.5px] top-3 w-px bg-divider/50" />
      <span className="absolute left-0 top-[5px]"><StatusDot tone={RUN_TONE[run.status]} /></span>
      <div className="flex items-baseline gap-2 text-xs">
        <span className="min-w-0 flex-1 truncate font-medium text-ink">{PHASE_LABELS[run.phase]}{run.round > 0 ? ` · round ${run.round}` : ""}</span>
        <span className={cn("shrink-0 text-[11px]", run.status === "failed" ? "text-danger" : "text-ink-subtle")}>{run.status}</span>
      </div>
      {run.summary && !hideSummary && <p className="mt-0.5 whitespace-pre-wrap break-words text-[11px] leading-relaxed text-ink-muted">{run.summary}</p>}
      <ListBlock label="Plan" items={run.phase === "plan" ? stringList(run.envelope?.steps) : []} />
      <ListBlock label="Findings" items={run.phase === "review" ? stringList(run.envelope?.findings) : []} />
      {question && <p className="mt-1 whitespace-pre-wrap text-[11px] text-ink">Question: {question}</p>}
      {run.gate_results.length > 0 && <ul className="mt-1 flex flex-col gap-1">{run.gate_results.map((gate, index) => <Gate key={index} gate={gate} />)}</ul>}
    </li>
  )
}

/** A question to the agent with the work it caused: the request or a later user message, then phase runs and agent replies. */
interface Turn {
  key: string
  question: string
  phases: PhaseRun[]
  replies: FactoryMessage[]
}

/** Group a task into turns: each user message opens one, phase runs and agent replies join the turn open when they started. */
function turnsOf(task: FactoryTaskDetail): Turn[] {
  const turns: (Turn & { since: number })[] = [{ key: "request", question: task.request, phases: [], replies: [], since: -Infinity }]
  for (const message of task.messages) {
    if (message.role === "user") turns.push({ key: `message-${message.id}`, question: message.text, phases: [], replies: [], since: Date.parse(message.created_at) })
    else turns[turns.length - 1].replies.push(message)
  }
  for (const run of task.phases) {
    const started = Date.parse(run.started_at)
    turns.findLast((turn) => turn.since <= started)!.phases.push(run)
  }
  return turns
}

const BLOCKED_PREFIX = "Blocked:"

/** An agent reply: a block reason as plain text in a bounded scroll area (engine errors run long), anything else as Markdown. */
function Reply({ message }: { message: FactoryMessage }) {
  if (message.text.startsWith(BLOCKED_PREFIX)) {
    return <p className="max-h-64 overflow-y-auto whitespace-pre-wrap break-words text-xs leading-relaxed text-danger scrollbar-thin">{message.text}</p>
  }
  return <LaTeXMarkdown content={message.text} />
}

function TurnView({ turn }: { turn: Turn }) {
  return (
    <TurnCard expanded>
      <TurnRegion role="question">
        <LaTeXMarkdown content={turn.question} />
      </TurnRegion>
      {turn.phases.length > 0 && (
        <TurnSurface>
          <TurnRegion role="answer" tail="activity" icon={<ListChecks size={14} aria-hidden="true" />}>
            <ol aria-label="Phases">
              {turn.phases.map((run) => <PhaseEntry key={run.id} run={run} hideSummary={turn.replies.some((reply) => reply.text.includes(run.summary))} />)}
            </ol>
          </TurnRegion>
        </TurnSurface>
      )}
      {turn.replies.map((reply) => (
        <TurnSurface key={reply.id}>
          <TurnRegion role="answer" actions={<CopyAction text={reply.text} label="Copy reply" />}>
            <Reply message={reply} />
          </TurnRegion>
        </TurnSurface>
      ))}
    </TurnCard>
  )
}

function MessageComposer({ taskId, disabled, onSent, onFailed }: { taskId: string; disabled: boolean; onSent: () => void; onFailed: (message: string) => void }) {
  const [text, setText] = useState("")
  const [sending, setSending] = useState(false)
  const { images, add: addImages, remove: removeImage, clear: clearImages, onPaste } = useImageAttachments(onFailed)

  const send = async () => {
    const trimmed = text.trim()
    if (!trimmed || sending || disabled) return
    setSending(true)
    try {
      await sendFactoryMessage(taskId, trimmed, toFactoryImages(images))
      setText("")
      clearImages()
      onSent()
    } catch (cause) {
      onFailed(errorMessage(cause, "Could not send the message"))
    } finally {
      setSending(false)
    }
  }

  return (
    <div className="shrink-0 p-3 pt-0">
      <Composer
        value={text}
        onValueChange={setText}
        onSubmit={() => void send()}
        onPaste={onPaste}
        label="Message to the agent"
        placeholder="Message"
        disabled={disabled || sending}
        leading={<AttachImagesButton disabled={disabled || sending} onPick={(files) => void addImages(files)} className={COMPOSER_ICON_BUTTON} />}
      >
        {images.length > 0 && <div className="pb-2"><StagedImages images={images} onRemove={removeImage} /></div>}
      </Composer>
    </div>
  )
}

const PRIMARY_ACTIONS = [
  { action: "start", label: "Start" },
  { action: "approve", label: "Approve" },
  { action: "retry", label: "Retry" },
] as const

const actionFailure = (action: TaskAction) => `Could not ${action} the task`

const PANEL_WIDTH_PX = 420

/**
 * A task's agent in the tab's file panel: state and its actions in the header, the request and later messages as turns
 * with their phase runs and replies, and the composer while the engine takes messages.
 */
export function AgentPanel({ task, onChanged, onClose }: { task: FactoryTaskDetail; onChanged: () => void; onClose: () => void }) {
  const [confirmingCancel, setConfirmingCancel] = useState(false)
  const { run, pending: busy, error, setError } = useAsyncAction(async (action: TaskAction) => {
    try {
      await runFactoryTaskAction(task.id, action)
      onChanged()
    } finally {
      setConfirmingCancel(false)
    }
  }, actionFailure)
  const scroller = useRef<HTMLDivElement>(null)
  const allowed = allowedActions(task.state)
  const turns = turnsOf(task)
  // The block message repeats the task error.
  const taskError = task.error && !task.messages.some((message) => message.text.includes(task.error!)) ? task.error : ""

  useEffect(() => {
    scroller.current?.scrollTo({ top: scroller.current.scrollHeight })
  }, [task.messages.length])

  const actions = <>
    <span className="min-w-0 flex-1 truncate">{STATE_LABELS[task.state]}</span>
    {PRIMARY_ACTIONS.map(({ action, label }) => allowed[action] && <button key={action} type="button" disabled={busy} onClick={() => void run(action)} className={PRIMARY_BUTTON_CLASS}>{label}</button>)}
    {allowed.cancel && !confirmingCancel && <button type="button" disabled={busy} onClick={() => setConfirmingCancel(true)} className={QUIET_BUTTON_CLASS}>Cancel</button>}
  </>

  return (
    <FilePanel label="Agent" bare onClose={onClose} width={PANEL_WIDTH_PX} actions={actions}>
      {confirmingCancel && <div className="px-4 pt-2"><InlineConfirm question="Cancel this task?" busy={busy} onYes={() => void run("cancel")} onNo={() => setConfirmingCancel(false)} /></div>}
      <ErrorText className="max-h-32 overflow-y-auto break-words px-4 pt-2 text-[11px]">{error || taskError}</ErrorText>
      <div ref={scroller} className="min-h-0 flex-1 overflow-y-auto scrollbar-thin">
        {turns.map((turn) => <TurnView key={turn.key} turn={turn} />)}
      </div>
      {allowed.message && <MessageComposer taskId={task.id} disabled={busy} onSent={onChanged} onFailed={setError} />}
    </FilePanel>
  )
}
