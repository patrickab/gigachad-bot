"use client"

import { memo, useState, useCallback } from "react"
import { ChevronDown, ChevronRight, Clock, Search, Globe, FileText, Bot, Zap, DollarSign, Layers, type LucideIcon } from "lucide-react"
import { toggled } from "@/lib/utils"
import type { ResearchTraceStep, ResearchTraceProgress } from "@/lib/domain/types"

interface Props {
  steps?: ResearchTraceStep[]
  progress?: ResearchTraceProgress
  isLive?: boolean
}

const STEPS: Record<string, { icon: LucideIcon; label: string }> = {
  start: { icon: Search, label: "Research started" },
  choose_agent: { icon: Bot, label: "Selecting agent" },
  agent_selected: { icon: Bot, label: "Agent selected" },
  conducting_research: { icon: Search, label: "Researching..." },
  deep_research_initialize: { icon: Layers, label: "Initializing deep research" },
  deep_research_start: { icon: Layers, label: "Deep research started" },
  research_completed: { icon: FileText, label: "Research completed" },
  deep_research_complete: { icon: FileText, label: "Deep research completed" },
  writing_report: { icon: FileText, label: "Writing report" },
  report_completed: { icon: FileText, label: "Report completed" },
  writing_introduction: { icon: FileText, label: "Writing introduction" },
  introduction_completed: { icon: FileText, label: "Introduction completed" },
  writing_conclusion: { icon: FileText, label: "Writing conclusion" },
  conclusion_completed: { icon: FileText, label: "Conclusion completed" },
  cost_update: { icon: DollarSign, label: "Cost update" },
  planning_images: { icon: FileText, label: "Planning images" },
  images_pre_generated: { icon: FileText, label: "Images generated" },
}

function stepIcon(step: string, eventType: string) {
  const Icon = STEPS[step]?.icon ?? (eventType === "tool" ? Zap : Globe)
  return <Icon className="h-3 w-3" />
}

function stepColor(eventType: string) {
  return ["research", "tool", "action"].includes(eventType) ? "text-ink bg-surface-elevated" : "text-ink-muted bg-hover"
}

function formatTime(ts: number) {
  return new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
}

/** Collapsible log of research events. Steps with `details` expand to raw JSON, and the live pulse stops once a terminal step arrives. */
function ResearchTraceInner({ steps = [], progress, isLive }: Props) {
  const [expanded, setExpanded] = useState(false)
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set())

  const toggleStep = useCallback((idx: number) => setExpandedSteps(prev => toggled(prev, idx)), [])

  if (steps.length === 0 && !progress && !isLive) return null

  // Any of these ends a run, depending on the research mode.
  const lastStep = steps[steps.length - 1]
  const isFinished = lastStep?.step === "report_completed" || lastStep?.step === "conclusion_completed" || lastStep?.step === "deep_research_complete"

  return (
    <div className="mt-2 rounded-md border border-divider bg-surface/60 overflow-hidden">
      <button
        onClick={() => setExpanded(e => !e)}
        className="w-full flex items-center gap-2 px-3 py-2 hover:bg-surface-elevated/50 transition-colors text-left"
      >
        {expanded ? (
          <ChevronDown className="h-3.5 w-3.5 text-ink-subtle shrink-0" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5 text-ink-subtle shrink-0" />
        )}
        <span className="text-[11px] font-medium text-ink-muted">
          Research Trace
        </span>
        <span className="text-[10px] text-ink-faint ml-auto tabular-nums">
          {steps.length} step{steps.length !== 1 ? "s" : ""}
        </span>
        {progress && (
          <span className="text-[10px] text-ink tabular-nums">
            {progress.completed_queries}/{progress.total_queries} queries
          </span>
        )}
        {isLive && !isFinished && (
          <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-ink ml-1" />
        )}
      </button>

      {expanded && (
        <div className="border-t border-divider/50 max-h-80 overflow-y-auto">
          {steps.map((step, idx) => {
            const isOpen = expandedSteps.has(idx)
            const hasDetails = step.details && Object.keys(step.details).length > 0
            const colorClass = stepColor(step.event_type)
            const isLast = idx === steps.length - 1

            return (
              <div key={idx} className="group">
                <button
                  onClick={() => hasDetails ? toggleStep(idx) : undefined}
                  className={`w-full flex items-center gap-2 px-3 py-1.5 text-left hover:bg-surface-elevated/30 transition-colors ${hasDetails ? "cursor-pointer" : "cursor-default"}`}
                >
                  <div className="flex items-center gap-1.5 min-w-0 flex-1">
                    <div className={`shrink-0 flex items-center justify-center h-5 w-5 rounded ${colorClass}`}>
                      {stepIcon(step.step, step.event_type)}
                    </div>
                    <span className="text-[11px] text-ink truncate flex-1">
                      {formatStepLabel(step.step, step.event_type)}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    {hasDetails && (
                      <ChevronDown className={`h-3 w-3 text-ink-faint transition-transform ${isOpen ? "rotate-180" : ""}`} />
                    )}
                      <span className="text-[9px] text-ink-faint font-mono tabular-nums flex items-center gap-1">
                      <Clock className="h-2.5 w-2.5" />
                      {formatTime(step.timestamp)}
                    </span>
                  </div>
                </button>
                {isOpen && hasDetails && (
                  <div className="px-3 pb-2 ml-7">
                    <pre className="text-[9px] text-ink-subtle bg-paper rounded p-2 overflow-x-auto whitespace-pre-wrap break-all">
                      {JSON.stringify(step.details, null, 2)}
                    </pre>
                  </div>
                )}
                {!isLast && (
                  <div className="ml-[1.375rem] h-px bg-surface-elevated/50" />
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

function formatStepLabel(step: string, eventType: string): string {
  return STEPS[step]?.label ?? (eventType === "tool" ? `Tool: ${step}` : step)
}

export const ResearchTrace = memo(ResearchTraceInner)