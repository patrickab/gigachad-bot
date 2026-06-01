"use client"

import { useCallback, useRef, useState } from "react"
import { createResearchStream } from "@/lib/api"
import type { Message, ResearchParams, ResearchTraceStep, ResearchTraceProgress } from "@/lib/types"

export type { ResearchParams }

export interface UseResearchReturn {
  research: (params: ResearchParams, appendMessage: (msg: Message) => void, updateLast: (msg: Message) => void) => Promise<void>
  error: string | null
}

export function useResearch(): UseResearchReturn {
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<(() => void) | null>(null)

  const research = useCallback(async (
    params: ResearchParams,
    appendMessage: (msg: Message) => void,
    updateLast: (msg: Message) => void,
  ) => {
    setError(null)

    const userMsg: Message = { role: "user", content: params.query }
    const assistantMsg: Message = { role: "assistant", content: "", research_steps: [], research_progress: undefined }

    appendMessage(userMsg)
    appendMessage(assistantMsg)

    try {
      const stream = createResearchStream({
        query: params.query,
        fast_model: params.fastModel,
        smart_model: params.smartModel,
        strategic_model: params.strategicModel,
        depth: params.depth,
        breadth: params.breadth,
        reasoning_effort: params.reasoningEffort === "none" ? null : params.reasoningEffort,
        report_type: params.reportType,
      })
      abortRef.current = stream.abort

      const steps: ResearchTraceStep[] = []
      let runId = ""

      for await (const event of stream) {
        try {
          const data = JSON.parse(event.data)

          if (event.event === "step") {
            steps.push({
              step: data.step,
              event_type: data.event_type,
              details: data.details || {},
              timestamp: data.timestamp || Date.now() / 1000,
            })
            assistantMsg.research_steps = [...steps]
            const stepLabel = formatStepLabel(data.step, data.event_type)
            assistantMsg.content = stepLabel
            updateLast(assistantMsg)
          } else if (event.event === "progress") {
            const progress: ResearchTraceProgress = {
              current_depth: data.current_depth ?? 0,
              total_depth: data.total_depth ?? 0,
              current_breadth: data.current_breadth ?? 0,
              total_breadth: data.total_breadth ?? 0,
              current_query: data.current_query ?? null,
              completed_queries: data.completed_queries ?? 0,
              total_queries: data.total_queries ?? 0,
            }
            assistantMsg.research_progress = progress
            const stepLabel = formatProgressLabel(progress)
            assistantMsg.content = stepLabel
            updateLast(assistantMsg)
          } else if (event.event === "result") {
            const result = data
            const sourcesList = result.sources?.length
              ? `\n\n---\n**Sources (${result.sources.length})** | **Cost**: $${Number(result.costs).toFixed(4)}\n${result.sources.map((s: string) => `- ${s}`).join("\n")}`
              : ""
            assistantMsg.content = result.report + sourcesList
            assistantMsg.research_trace_id = runId || undefined
            updateLast(assistantMsg)
          } else if (event.event === "error") {
            throw new Error(data.message || "Research failed")
          }
        } catch (parseErr) {
          if ((parseErr as Error).message === "Research failed" || (parseErr as Error)?.message?.includes("Research failed")) {
            throw parseErr
          }
        }
      }
    } catch (e) {
      if ((e as Error).name === "AbortError") return
      const msg = (e as Error)?.message ?? "Research failed"
      setError(msg)
      assistantMsg.content = assistantMsg.content || `Research error: ${msg}`
      updateLast(assistantMsg)
    } finally {
      abortRef.current = null
    }
  }, [])

  return { research, error }
}

function formatStepLabel(step: string, eventType: string): string {
  const labels: Record<string, string> = {
    start: "🔍 Starting research...",
    choose_agent: "🤖 Selecting research agent...",
    agent_selected: "🤖 Agent selected",
    conducting_research: "🔎 Conducting research...",
    deep_research_initialize: "🧠 Initializing deep research...",
    deep_research_start: "🚀 Starting deep research",
    research_completed: "✅ Research completed",
    deep_research_complete: "✅ Deep research completed",
    writing_report: "📝 Writing report...",
    report_completed: "✅ Report written",
    writing_introduction: "📖 Writing introduction...",
    introduction_completed: "✅ Introduction written",
    writing_conclusion: "✍️ Writing conclusion...",
    conclusion_completed: "✅ Conclusion written",
    cost_update: "💰 Updating costs...",
    planning_images: "🖼️ Planning images...",
    images_pre_generated: "✅ Images generated",
  }
  if (labels[step]) return labels[step]
  if (eventType === "tool") return `🔧 Running tool: ${step}`
  if (eventType === "action") return `⚡ ${step}`
  return `⏳ ${step}...`
}

function formatProgressLabel(progress: ResearchTraceProgress): string {
  const parts: string[] = []
  if (progress.total_depth > 0) {
    parts.push(`Depth ${progress.current_depth}/${progress.total_depth}`)
  }
  if (progress.total_queries > 0) {
    parts.push(`${progress.completed_queries}/${progress.total_queries} queries`)
  }
  if (progress.current_query) {
    const q = progress.current_query.length > 60 ? progress.current_query.slice(0, 57) + "..." : progress.current_query
    parts.push(`🔍 ${q}`)
  }
  return parts.join(" · ") || "Researching..."
}