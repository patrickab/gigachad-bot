"use client"

import { useId, useState, type ReactNode } from "react"
import { Check, ChevronDown, CircleAlert, CircleSlash, Factory, LoaderCircle, type LucideIcon } from "lucide-react"
import { Disclosure } from "@/components/ui/Disclosure"
import { isRecord } from "@/lib/utils"
import { registerToolRenderer } from "./renderers"
import { PairedResult } from "./tools/DiffView"
import { formatDuration } from "./tools/format"
import type { TextContent, ToolRun } from "./types"
import "@/features/conversation/tools/tools.css"

/** The text blocks of a run's result joined, null before it has one. */
export function resultText(run: ToolRun | undefined): string | null {
  if (!run?.result) return null
  return run.result.content.filter((block): block is TextContent => block.type === "text").map((block) => block.text).join("\n")
}

interface ActivityRowProps {
  run: ToolRun | undefined
  icon: LucideIcon
  tool: string
  summary: string
  /** At the right of a finished row: a status line, else the duration. */
  status?: string
  children: ReactNode
}

/** The default tool row's header (`tools/ToolCallRow.tsx`) with a custom summary and details; collapsed at first. */
export function ActivityRow({ run, icon: Icon, tool, summary, status, children }: ActivityRowProps) {
  const [open, setOpen] = useState(false)
  const detailsId = useId()
  const isRunning = run?.running === true
  const isError = run?.isError === true
  const isDone = run !== undefined && !isRunning
  return (
    <div className="pi-activity-row" data-activity-operation="true">
      <div className="pi-activity-row-header" onClick={() => setOpen(!open)}>
        <button type="button" className="pi-activity-row-trigger" data-fill="true" aria-expanded={open} aria-controls={detailsId}>
          <span className="pi-activity-row-indicator" data-error={isError || undefined} aria-hidden>
            {isError ? (
              <CircleAlert size={12} strokeWidth={1.8} />
            ) : isDone ? (
              <Check size={12} strokeWidth={2} />
            ) : isRunning ? (
              <LoaderCircle size={12} strokeWidth={1.8} className="pi-activity-row-spinner" />
            ) : (
              <CircleSlash size={12} strokeWidth={1.8} className="pi-activity-row-idle" />
            )}
          </span>
          <span className="sr-only">{isError ? "failed" : isDone ? "done" : isRunning ? "running" : "not run"}</span>
          <span className="pi-activity-tool-icon" aria-hidden><Icon size={12} strokeWidth={1.8} /></span>
          <span className="pi-activity-row-tool" data-error={isError || undefined}>{tool}</span>
          <span className="pi-activity-row-preview">{summary}</span>
        </button>
        {isDone && status ? (
          <span className={`pi-activity-row-duration max-w-[45%] truncate ${isError ? "text-danger" : ""}`}>{status}</span>
        ) : isDone && run.durationMs !== undefined ? (
          <span className="pi-activity-row-duration">{formatDuration(run.durationMs)}</span>
        ) : null}
        <ChevronDown size={12} strokeWidth={1.8} aria-hidden className="pi-activity-row-chevron" data-expanded={open || undefined} />
      </div>
      {open && (
        <div id={detailsId} className="pi-tool-call-details space-y-1 px-2 py-1.5 text-[11px]" data-error={isError || undefined}>
          {children}
        </div>
      )}
    </div>
  )
}

/** The run's result behind a collapsed `Result` disclosure: plain text, or `render` of it. */
export function ResultDisclosure({ run, render }: { run: ToolRun | undefined; render?: (text: string) => ReactNode }) {
  const text = resultText(run)
  if (text === null || run?.running) return null
  const isEmpty = text.trim() === ""
  return (
    <Disclosure summary="Result">
      {render && !isEmpty && !run?.isError ? render(text) : <PairedResult text={text} isEmpty={isEmpty} isError={run?.isError === true} />}
    </Disclosure>
  )
}

const VERBS: Record<string, string> = {
  "waves.start": "start wave", "waves.add": "add tasks", "waves.get": "wave", "waves.list": "waves",
  "waves.timeline": "wave timeline", "waves.retry": "retry wave", "waves.approve": "approve wave",
  "waves.cancel": "cancel wave", "waves.remove": "remove wave",
  "tasks.start": "start task", "tasks.message": "message", "tasks.get": "task", "tasks.list": "tasks",
  "tasks.wait": "wait", "tasks.timeline": "timeline", "tasks.diff": "diff", "tasks.import": "import task",
  "tasks.retry": "retry task", "tasks.cancel": "cancel task", "tasks.approve": "approve task", "tasks.remove": "remove task",
  "settings.get": "settings", "settings.set": "set settings", "models.get": "models", "models.set": "set models",
  "config.get": "config", "config.write": "write config", "repos.list": "repos", "repos.add": "add repo",
  "repos.remove": "remove repo", "branches.list": "branches",
}

/** Long text fields shown behind a disclosure of their own, with their label per action. */
function longTextLabel(action: string, key: string): string | null {
  if (key === "request") return action.startsWith("waves.") ? "Reviewer instructions" : "Request"
  if (key === "text") return action === "tasks.message" ? "Message" : action === "config.write" ? "factory.yaml" : "Text"
  return null
}

interface WaveTask { key: string; title: string; model?: unknown; after?: unknown }

const isWaveTask = (value: unknown): value is WaveTask => isRecord(value) && typeof value.key === "string" && typeof value.title === "string"
const shortId = (value: unknown): string | undefined => (typeof value === "string" && value ? value.slice(0, 8) : undefined)
const plural = (count: number, noun: string) => `${count} ${noun}${count === 1 ? "" : "s"}`

/** A parameter's value as one line: lists joined, objects as `key: value` pairs. */
function formatValue(value: unknown): string {
  if (value === null || value === undefined) return ""
  if (Array.isArray(value)) return value.map(formatValue).join(", ")
  if (isRecord(value)) return Object.entries(value).map(([key, inner]) => `${key}: ${formatValue(inner)}`).join(", ")
  return String(value)
}

/** The collapsed row's text after the verb: the tag, title or short id the call names, and a wave's task count. */
function factorySummary(action: string, params: Record<string, unknown>, tasks: WaveTask[] | null): string {
  const verb = VERBS[action] ?? action
  const ids = Array.isArray(params.ids) ? params.ids.map(shortId).filter(Boolean).join(", ") : undefined
  const named = [params.tag, params.title].find((value): value is string => typeof value === "string" && value !== "")
    ?? shortId(params.id) ?? shortId(params.wave) ?? (ids || undefined)
  return [verb, named, tasks ? plural(tasks.length, "task") : undefined].filter(Boolean).join(" · ")
}

function WaveTaskLines({ tasks }: { tasks: WaveTask[] }) {
  return (
    <div className="grid grid-cols-[auto_auto_minmax(0,1fr)_auto] gap-x-3 text-ink-muted">
      {tasks.map((task) => {
        const after = Array.isArray(task.after) ? task.after.filter((key) => typeof key === "string") : []
        return (
          <div key={task.key} className="contents" data-wave-task={task.key}>
            <span className="text-ink">{task.key}</span>
            <span>{typeof task.model === "string" && task.model ? task.model : "big"}</span>
            <span className="truncate">{task.title}</span>
            <span className="text-ink-subtle">{after.length > 0 ? `after ${after.join(", ")}` : ""}</span>
          </div>
        )
      })}
    </div>
  )
}

/**
 * The `factory` tool's row: a verb for the action and what it names, the result's first line at the right; opened,
 * the parameters as a short list, a wave's tasks one line each, long text and the result behind disclosures.
 */
registerToolRenderer("factory", (call, run, next) => {
  const { action, ...params } = isRecord(call.arguments) ? call.arguments : {}
  if (typeof action !== "string" || action === "") return next()
  if (params.tasks !== undefined && !(Array.isArray(params.tasks) && params.tasks.every(isWaveTask))) return next()
  const tasks = Array.isArray(params.tasks) ? (params.tasks as WaveTask[]) : null
  const firstLine = resultText(run)?.trim().split("\n")[0]
  const fields = Object.entries(params).filter(([key, value]) => key !== "title" && key !== "tasks" && longTextLabel(action, key) === null && formatValue(value) !== "")
  const longTexts = Object.entries(params).flatMap(([key, value]) => {
    const label = longTextLabel(action, key)
    return label && typeof value === "string" && value.trim() ? [{ key, label, value }] : []
  })
  return (
    <ActivityRow run={run} icon={Factory} tool="factory" summary={factorySummary(action, params, tasks)} status={firstLine}>
      {typeof params.title === "string" && params.title && <p className="text-ink">{params.title}</p>}
      {tasks && tasks.length > 0 && <WaveTaskLines tasks={tasks} />}
      {fields.length > 0 && (
        <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 text-ink-muted">
          {fields.map(([key, value]) => (
            <div key={key} className="contents">
              <dt className="text-ink-subtle">{key}</dt>
              <dd className="m-0 break-words">{formatValue(value)}</dd>
            </div>
          ))}
        </dl>
      )}
      {longTexts.map(({ key, label, value }) => (
        <Disclosure key={key} summary={label}>
          <pre className="m-0 whitespace-pre-wrap break-words pl-5 font-sans text-xs text-ink-muted">{value}</pre>
        </Disclosure>
      ))}
      <ResultDisclosure run={run} />
    </ActivityRow>
  )
})
