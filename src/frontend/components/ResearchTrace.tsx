"use client"

import { memo, useState, useCallback } from "react"
import { ChevronDown, ChevronRight, Clock, Search, Globe, FileText, Bot, Zap, DollarSign, Layers } from "lucide-react"
import type { ResearchTraceStep, ResearchTraceProgress } from "@/lib/types"

interface Props {
  steps?: ResearchTraceStep[]
  progress?: ResearchTraceProgress
  traceId?: string
  isLive?: boolean
}

const STEP_ICONS: Record<string, React.ReactNode> = {
  start: <Search className="h-3 w-3" />,
  choose_agent: <Bot className="h-3 w-3" />,
  agent_selected: <Bot className="h-3 w-3" />,
  conducting_research: <Search className="h-3 w-3" />,
  deep_research_initialize: <Layers className="h-3 w-3" />,
  deep_research_start: <Layers className="h-3 w-3" />,
  research_completed: <FileText className="h-3 w-3" />,
  deep_research_complete: <FileText className="h-3 w-3" />,
  writing_report: <FileText className="h-3 w-3" />,
  report_completed: <FileText className="h-3 w-3" />,
  writing_introduction: <FileText className="h-3 w-3" />,
  introduction_completed: <FileText className="h-3 w-3" />,
  writing_conclusion: <FileText className="h-3 w-3" />,
  conclusion_completed: <FileText className="h-3 w-3" />,
  cost_update: <DollarSign className="h-3 w-3" />,
  planning_images: <FileText className="h-3 w-3" />,
  images_pre_generated: <FileText className="h-3 w-3" />,
}

const TYPE_COLORS: Record<string, string> = {
  research: "text-ink bg-surface-elevated",
  tool: "text-ink bg-surface-elevated",
  action: "text-ink bg-surface-elevated",
}

function stepIcon(step: string, eventType: string) {
  if (STEP_ICONS[step]) return STEP_ICONS[step]
  if (eventType === "tool") return <Zap className="h-3 w-3" />
  return <Globe className="h-3 w-3" />
}

function stepColor(eventType: string) {
  return TYPE_COLORS[eventType] || "text-ink-muted bg-hover"
}

function formatTime(ts: number) {
  return new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })
}

function ResearchTraceInner({ steps = [], progress, traceId, isLive }: Props) {
  const [expanded, setExpanded] = useState(false)
  const [expandedSteps, setExpandedSteps] = useState<Set<number>>(new Set())

  const toggleStep = useCallback((idx: number) => {
    setExpandedSteps(prev => {
      const next = new Set(prev)
      if (next.has(idx)) next.delete(idx)
      else next.add(idx)
      return next
    })
  }, [])

  if (steps.length === 0 && !progress && !isLive) return null

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
        <span className="text-[10px] text-ink-faint ml-auto">
          {steps.length} step{steps.length !== 1 ? "s" : ""}
        </span>
        {progress && (
          <span className="text-[10px] text-ink">
            {progress.completed_queries}/{progress.total_queries} queries
          </span>
        )}
        {isLive && !isFinished && (
          <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-ink ml-1" />
        )}
        {traceId && (
          <span className="text-[9px] text-ink-faint font-mono ml-1">#{traceId}</span>
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
                    <span className="text-[9px] text-ink-faint font-mono flex items-center gap-1">
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
  const labels: Record<string, string> = {
    start: "Research started",
    choose_agent: "Selecting agent",
    agent_selected: "Agent selected",
    conducting_research: "Researching...",
    deep_research_initialize: "Initializing deep research",
    deep_research_start: "Deep research started",
    research_completed: "Research completed",
    deep_research_complete: "Deep research completed",
    writing_report: "Writing report",
    report_completed: "Report completed",
    writing_introduction: "Writing introduction",
    introduction_completed: "Introduction completed",
    writing_conclusion: "Writing conclusion",
    conclusion_completed: "Conclusion completed",
    cost_update: "Cost update",
    planning_images: "Planning images",
    images_pre_generated: "Images generated",
  }
  if (labels[step]) return labels[step]
  if (eventType === "tool") return `Tool: ${step}`
  if (eventType === "action") return step
  return step
}

export const ResearchTrace = memo(ResearchTraceInner)