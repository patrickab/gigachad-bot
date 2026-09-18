"use client"

import { memo, useState } from "react"
import { ChevronDown, Globe, LineChart, Loader2, Search, Wrench } from "lucide-react"
import { cn } from "@/lib/utils"
import type { ToolCallRecord } from "@/lib/types"
import { PlotElement, type PlotFigure } from "./PlotElement"

const TOOL_META: Record<string, { label: string; icon: typeof Globe }> = {
  web_search: { label: "Web search", icon: Globe },
  deep_research: { label: "Deep research", icon: Search },
  plot: { label: "Plot", icon: LineChart },
}

interface ToolCallElementProps {
  call: ToolCallRecord
}

function primaryArgument(args: Record<string, unknown>): string {
  const value = args.query ?? Object.values(args)[0]
  return typeof value === "string" ? value : ""
}

/** Sits between the question and the answer, wearing the same container optic as the
 *  assistant response: same row layout, same avatar, same type scale. A `plot` call renders
 *  its figure directly below the header, not gated behind expand — the figure is the point. */
function ToolCallElementInner({ call }: ToolCallElementProps) {
  const [open, setOpen] = useState(false)
  const isPlot = call.name === "plot"
  const meta = TOOL_META[call.name] ?? { label: call.name, icon: Wrench }
  const Icon = meta.icon
  const argument = isPlot ? "" : primaryArgument(call.arguments)
  const sources = call.sources ?? []
  const costs = typeof call.detail?.costs === "number" ? (call.detail.costs as number) : null
  const figure = isPlot ? (call.detail?.figure as PlotFigure | undefined) : undefined
  const status = call.status === "running" ? "running" : call.status === "error" ? "failed" : call.summary

  return (
    <div className="text-ink">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-start gap-3 px-6 py-5 text-left"
      >
        <div className="mt-0.5 shrink-0">
          <div className="flex h-7 w-7 items-center justify-center rounded-xl bg-surface-elevated">
            {call.status === "running"
              ? <Loader2 className="h-3.5 w-3.5 animate-spin text-ink-muted" aria-hidden="true" />
              : <Icon className={cn("h-3.5 w-3.5", call.status === "error" ? "text-danger" : "text-ink")} aria-hidden="true" />}
          </div>
        </div>
        <div className="min-w-0 flex-1 flex flex-col">
          <div className="mb-0.5 text-xs font-medium text-ink-subtle">{meta.label}</div>
          {argument && <span className="truncate text-sm text-ink">{argument}</span>}
        </div>
        {status && <span className="mt-1 shrink-0 text-[10px] tabular-nums text-ink-faint">{status}</span>}
        <ChevronDown className={cn("mt-1 h-4 w-4 shrink-0 text-ink-faint transition-transform", open && "rotate-180")} aria-hidden="true" />
      </button>

      {figure && (
        <div className="px-6 pb-5 pl-[3.25rem]">
          <PlotElement figure={figure} />
        </div>
      )}

      {open && (
        <div className="space-y-2 px-6 pb-5 pl-[4.5rem]">
          {isPlot ? (
            typeof call.arguments.code === "string" && (
              <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-words rounded-md bg-surface-elevated p-2 text-[10px] text-ink-muted">
                {call.arguments.code}
              </pre>
            )
          ) : (
            <dl className="space-y-1">
              {Object.entries(call.arguments).map(([key, value]) => (
                <div key={key} className="flex gap-2 text-[10px]">
                  <dt className="shrink-0 font-medium uppercase tracking-wide text-ink-faint">{key}</dt>
                  <dd className="min-w-0 break-words text-ink-muted">{typeof value === "string" ? value : JSON.stringify(value)}</dd>
                </div>
              ))}
            </dl>
          )}

          {call.error && <p className="text-[10px] text-danger">{call.error}</p>}

          {costs !== null && <p className="text-[10px] tabular-nums text-ink-faint">cost ${costs.toFixed(4)}</p>}

          {sources.length > 0 && (
            <ul className="space-y-1">
              {sources.map((source) => (
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
        </div>
      )}
    </div>
  )
}

export const ToolCallElement = memo(ToolCallElementInner)
