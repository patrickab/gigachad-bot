"use client"

import { memo, useEffect, useState, type ReactNode } from "react"
import { ChevronDown, Loader2 } from "lucide-react"
import { cn } from "@/lib/utils"
import { toolDisplay } from "@/features/chat/tools"
import { toolPayload } from "@/features/chat/toolCalls"
import type { ToolCallRecord } from "@/lib/domain/types"
import { CodeBlock } from "@/components/content/CodeBlock"
import { PlotElement } from "@/components/content/PlotElement"
import { SandboxOutputElement } from "./SandboxOutputElement"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"

interface ToolCallElementProps {
  call: ToolCallRecord
}

/** Header text for a call: the `query` argument, else the first argument, if it is a string. */
function primaryArgument(args: Record<string, unknown>): string {
  const value = args.query ?? Object.values(args)[0]
  return typeof value === "string" ? value : ""
}

/** Elapsed time of a stage. A running stage ticks locally from `started_at` (epoch seconds), never below the server-reported duration. */
function stageDuration(stage: NonNullable<ToolCallRecord["stages"]>[number], now: number): string {
  const seconds = stage.status === "running"
    ? Math.max(stage.duration, now / 1_000 - stage.started_at)
    : stage.duration
  return `${seconds.toFixed(1)}s`
}

/** A text-button toggle that reveals `children` below it. */
function Disclosure({ label, children }: { label: string; children: ReactNode }) {
  const [open, setOpen] = useState(false)
  return <>
    <button
      type="button"
      onClick={() => setOpen((isOpen) => !isOpen)}
      aria-expanded={open}
      className="flex items-center gap-1 text-xs font-medium text-ink-subtle"
    >
      <ChevronDown className={cn("h-4 w-4 transition-transform", open && "rotate-180")} aria-hidden="true" />
      <span>{label}</span>
    </button>
    {open && children}
  </>
}

/** Sits between the question and the answer, wearing the same container optic as the
 *  assistant response: same row layout, same avatar, same type scale. */
function ToolCallElementInner({ call }: ToolCallElementProps) {
  const [open, setOpen] = useState(false)
  const shown = toolPayload(call)
  const [now, setNow] = useState(() => Date.now())
  const stages = call.stages ?? []
  const runningStage = stages.find((stage) => stage.status === "running")
  // Re-render at 10 Hz only while a stage runs so its timer ticks.
  useEffect(() => {
    if (!runningStage) return
    const timer = window.setInterval(() => setNow(Date.now()), 100)
    return () => window.clearInterval(timer)
  }, [runningStage])
  const { icon: Icon, label } = toolDisplay(call.name)
  // A plot, diagram, or mind map is its own headline, so the header carries no argument.
  const argument = shown.family === "plot" || shown.family === "diagram" || shown.family === "mindmap" ? "" : primaryArgument(call.arguments)
  // A running stage already names itself inline beside the card label.
  const status = call.status === "running" ? (stages.length > 0 ? "" : "running") : call.status === "error" ? "failed" : call.summary

  const stageTimeline = stages.length > 0 && (
    <span className="flex min-w-0 items-center gap-x-3 truncate text-[10px]">
      {stages.map((stage) => (
        <span key={stage.id} className="flex shrink-0 items-center gap-1.5">
          {stage.status === "running" && <Loader2 className="h-3 w-3 animate-spin text-ink-faint" aria-hidden="true" />}
          <span className="tabular-nums text-ink-faint">{stageDuration(stage, now)}</span>
          <span className={stage.status === "running" ? "text-ink" : stage.status === "error" ? "text-danger" : "text-ink-muted"}>
            {stage.status === "error" ? "Failed" : stage.label}
          </span>
        </span>
      ))}
    </span>
  )
  const header = <>
    <div className="shrink-0">
      <div className="flex h-7 w-7 items-center justify-center rounded-xl border border-divider bg-surface-elevated">
        <Icon className={cn("h-3.5 w-3.5", call.status === "error" ? "text-danger" : "text-ink")} aria-hidden="true" />
      </div>
    </div>
    <div className="min-w-0 flex-1 flex flex-col">
      <div className="flex min-h-7 min-w-0 items-center gap-x-4">
        <span className="shrink-0 text-xs font-medium text-ink-subtle">{label}</span>
        {stageTimeline}
      </div>
      {argument && <span className="mt-0.5 truncate text-sm text-ink">{argument}</span>}
    </div>
    {status && <span className="mt-1 shrink-0 text-[10px] tabular-nums text-ink-faint">{status}</span>}
  </>

  // Plots and diagrams are their own headline, so they have no disclosure.
  const flat = shown.family === "plot" || shown.family === "diagram"
  const inlineMarkdown = shown.family === "mindmap"
    ? shown.content
    : shown.family === "diagram" && shown.mermaid ? `\`\`\`mermaid\n${shown.mermaid}\n\`\`\`` : ""

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
      {flat ? (
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

      {(shown.family === "mindmap" || shown.family === "diagram") && (
        <div className="px-6 pb-5 pl-[3.25rem]">
          {inlineMarkdown ? <LaTeXMarkdown content={inlineMarkdown} /> : call.error ? <p className="text-xs text-danger">{call.error}</p> : null}
        </div>
      )}
      {shown.family === "plot" && <>
        {shown.brief && (
          <div className="px-6 pb-3 pl-[3.25rem]">
            <Disclosure label="Briefing">
              <p className="mt-2 whitespace-pre-line text-sm text-ink-muted">{shown.brief}</p>
            </Disclosure>
          </div>
        )}
        {shown.figure && (
          <div className="px-6 pb-5 pl-[3.25rem]">
            <PlotElement figure={shown.figure} />
          </div>
        )}
        {shown.script && (
          <div className="px-6 pb-3 pl-[3.25rem]">
            <Disclosure label="Code">
              <div className="mt-2"><CodeBlock codeString={shown.script} language="python" /></div>
            </Disclosure>
          </div>
        )}
        {/* Plots and diagrams have no disclosure, so a failed run states its error inline. */}
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

      {open && !flat && (
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
