"use client"

import { memo, useEffect, useState } from "react"
import { ChevronDown, Loader2, Wrench } from "lucide-react"
import { cn } from "@/lib/utils"
import { TOOL_META } from "@/hooks/useModeState"
import type { PlotFigure, SandboxToolResultRecord, ToolCallRecord, ToolSource } from "@/lib/types"
import { CodeBlock } from "./CodeBlock"
import { PlotElement } from "./PlotElement"
import { SandboxOutputElement } from "./SandboxOutputElement"

interface ToolCallElementProps {
  call: ToolCallRecord
}

/** How a call is drawn once its name has been read. Four families cover every tool: `sources`
 *  cites what it read, `plot` shows a figure, `workspace` reports a sandbox run, and `generic`
 *  catches a name this build no longer knows (an older saved chat). */
type Presentation =
  | { family: "sources"; sources: ToolSource[]; costs: number | null }
  | { family: "plot"; figure: PlotFigure | null; brief: string | null; script: string | null }
  | { family: "workspace"; sandbox: SandboxToolResultRecord | null }
  | { family: "generic" }

/** The only place a tool name is read. Another citing tool joins the `sources` case; nothing
 *  below this switch branches on names. */
function presentationOf(call: ToolCallRecord): Presentation {
  switch (call.name) {
    case "web_search":
    case "deep_research":
      return {
        family: "sources",
        sources: call.sources ?? [],
        costs: typeof call.detail?.costs === "number" ? call.detail.costs : null,
      }
    case "sandbox_plot":
      return {
        family: "plot",
        figure: call.detail?.figure ?? null,
        brief: call.detail?.brief ?? null,
        script: call.detail?.script ?? null,
      }
    case "workspace_agent":
      return { family: "workspace", sandbox: call.sandbox ?? null }
    default:
      return { family: "generic" }
  }
}

function primaryArgument(args: Record<string, unknown>): string {
  const value = args.query ?? Object.values(args)[0]
  return typeof value === "string" ? value : ""
}

function stageDuration(stage: NonNullable<ToolCallRecord["stages"]>[number], now: number): string {
  const seconds = stage.status === "running"
    ? Math.max(stage.duration, now / 1_000 - stage.started_at)
    : stage.duration
  return `${seconds.toFixed(1)}s`
}

/** Sits between the question and the answer, wearing the same container optic as the
 *  assistant response: same row layout, same avatar, same type scale. */
function ToolCallElementInner({ call }: ToolCallElementProps) {
  const [open, setOpen] = useState(false)
  const [briefOpen, setBriefOpen] = useState(false)
  const [codeOpen, setCodeOpen] = useState(false)
  const shown = presentationOf(call)
  const [now, setNow] = useState(() => Date.now())
  const stages = call.stages ?? []
  const runningStage = stages.find((stage) => stage.status === "running")
  useEffect(() => {
    if (!runningStage) return
    const timer = window.setInterval(() => setNow(Date.now()), 100)
    return () => window.clearInterval(timer)
  }, [runningStage])
  // Unknown saved names have no metadata; they keep the raw name and a neutral icon.
  const meta = TOOL_META[call.name]
  const Icon = meta?.icon ?? Wrench
  // A plot's figure is its headline, so the header carries neither argument nor disclosure.
  const argument = shown.family === "plot" ? "" : primaryArgument(call.arguments)
  const status = runningStage?.label ?? (call.status === "running" ? "running" : call.status === "error" ? "failed" : call.summary)

  const header = <>
    <div className="mt-0.5 shrink-0">
      <div className="flex h-7 w-7 items-center justify-center rounded-xl bg-surface-elevated">
        {call.status === "running"
          ? <Loader2 className="h-3.5 w-3.5 animate-spin text-ink-muted" aria-hidden="true" />
          : <Icon className={cn("h-3.5 w-3.5", call.status === "error" ? "text-danger" : "text-ink")} aria-hidden="true" />}
      </div>
    </div>
    <div className="min-w-0 flex-1 flex flex-col">
      <div className="mb-0.5 text-xs font-medium text-ink-subtle">{meta?.cardLabel ?? call.name}</div>
      {argument && <span className="truncate text-sm text-ink">{argument}</span>}
    </div>
    {status && <span className="mt-1 shrink-0 text-[10px] tabular-nums text-ink-faint">{status}</span>}
  </>

  const stageTimeline = stages.length > 0 && (
    <ol className="space-y-1 border-t border-divider/30 px-6 py-3 pl-[4.5rem]">
      {stages.map((stage) => (
        <li key={stage.id} className="flex items-center gap-2 text-[10px]">
          <span className="flex h-3 w-3 shrink-0 items-center justify-center">
            {stage.status === "running" && <Loader2 className="h-3 w-3 animate-spin text-ink-faint" aria-hidden="true" />}
          </span>
          <span className="shrink-0 tabular-nums text-ink-faint">{stageDuration(stage, now)}</span>
          <span className={cn("truncate", stage.status === "running" ? "text-ink" : stage.status === "error" ? "text-danger" : "text-ink-muted")}>
            {stage.status === "error" ? "Failed" : stage.label}
          </span>
        </li>
      ))}
    </ol>
  )

  const familyDetails = shown.family === "sources" ? <>
    {shown.costs !== null && <p className="text-[10px] tabular-nums text-ink-faint">cost ${shown.costs.toFixed(4)}</p>}
    {shown.sources.length > 0 && (
      <ul className="space-y-1">
        {shown.sources.map((source) => (
          <li key={source.url} className="flex min-w-0 items-baseline gap-2 text-[10px]">
            {source.label && <span className="shrink-0 font-medium text-ink-subtle">[{source.label}]</span>}
            <a
              href={source.url}
              target="_blank"
              rel="noopener noreferrer"
              className="min-w-0 truncate text-ink-muted underline decoration-divider-strong hover:text-ink"
            >
              {source.title || source.url}
            </a>
          </li>
        ))}
      </ul>
    )}
  </> : null

  return (
    <div className="text-ink">
      {shown.family === "plot" ? (
        <div className="flex w-full items-start gap-3 px-6 py-5 text-left">
          {header}
        </div>
      ) : (
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          className="flex w-full items-start gap-3 px-6 py-5 text-left"
        >
          {header}
          <ChevronDown className={cn("mt-1 h-4 w-4 shrink-0 text-ink-faint transition-transform", open && "rotate-180")} aria-hidden="true" />
        </button>
      )}

      {stageTimeline}

      {shown.family === "plot" && <>
        {shown.brief && (
          <div className="px-6 pb-3 pl-[3.25rem]">
            <button
              type="button"
              onClick={() => setBriefOpen((isOpen) => !isOpen)}
              aria-expanded={briefOpen}
              className="flex items-center gap-1 text-xs font-medium text-ink-subtle"
            >
              <ChevronDown className={cn("h-4 w-4 transition-transform", briefOpen && "rotate-180")} aria-hidden="true" />
              <span>Briefing</span>
            </button>
            {briefOpen && <p className="mt-2 whitespace-pre-line text-sm text-ink-muted">{shown.brief}</p>}
          </div>
        )}
        {shown.figure && (
          <div className="px-6 pb-5 pl-[3.25rem]">
            <PlotElement figure={shown.figure} />
          </div>
        )}
        {shown.script && (
          <div className="px-6 pb-3 pl-[3.25rem]">
            <button
              type="button"
              onClick={() => setCodeOpen((isOpen) => !isOpen)}
              aria-expanded={codeOpen}
              className="flex items-center gap-1 text-xs font-medium text-ink-subtle"
            >
              <ChevronDown className={cn("h-4 w-4 transition-transform", codeOpen && "rotate-180")} aria-hidden="true" />
              <span>Code</span>
            </button>
            {codeOpen && <div className="mt-2"><CodeBlock codeString={shown.script} language="python" /></div>}
          </div>
        )}
        {/* The plot header has no disclosure, so a failed run states its error inline. */}
        {call.error && <p className="px-6 pb-5 pl-[3.25rem] text-xs text-danger">{call.error}</p>}
      </>}

      {/* Outputs describe a finished run; while running the header spinner is the whole story. */}
      {shown.family === "workspace" && call.status !== "running" && (
        <div className="px-6 pb-5 pl-[3.25rem] space-y-3">
          {shown.sandbox && shown.sandbox.outputs.length > 0 ? (
            shown.sandbox.outputs.map((output, i) => <SandboxOutputElement key={output.display_id ?? i} output={output} />)
          ) : shown.sandbox?.status === "failed" || call.status === "error" ? (
            <p className="text-xs text-danger">{call.error ?? "The workspace agent run failed."}</p>
          ) : (
            <p className="text-xs text-ink-faint">Workspace updated.</p>
          )}
        </div>
      )}

      {open && shown.family !== "plot" && (
        <div className="space-y-2 px-6 pb-5 pl-[4.5rem]">
          <dl className="space-y-1">
            {Object.entries(call.arguments).map(([key, value]) => (
              <div key={key} className="flex gap-2 text-[10px]">
                <dt className="shrink-0 font-medium uppercase tracking-wide text-ink-faint">{key}</dt>
                <dd className="min-w-0 break-words text-ink-muted">{typeof value === "string" ? value : JSON.stringify(value)}</dd>
              </div>
            ))}
          </dl>

          {call.error && <p className="text-[10px] text-danger">{call.error}</p>}

          {familyDetails}
        </div>
      )}
    </div>
  )
}

export const ToolCallElement = memo(ToolCallElementInner)
