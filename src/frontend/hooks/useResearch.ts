"use client"

import { useCallback, useState } from "react"
import { runResearch } from "@/lib/api"
import type { Message } from "@/lib/types"

export interface ResearchParams {
  query: string
  fastModel: string
  smartModel: string
  strategicModel: string
  depth: number
  breadth: number
  reasoningEffort: string
  reportType: string
}

export interface UseResearchReturn {
  research: (params: ResearchParams, appendMessage: (msg: Message) => void, updateLast: (msg: Message) => void) => Promise<void>
  error: string | null
}

export function useResearch(): UseResearchReturn {
  const [error, setError] = useState<string | null>(null)

  const research = useCallback(async (
    params: ResearchParams,
    appendMessage: (msg: Message) => void,
    updateLast: (msg: Message) => void,
  ) => {
    setError(null)

    const userMsg: Message = { role: "user", content: params.query }
    const assistantMsg: Message = { role: "assistant", content: "" }

    appendMessage(userMsg)
    appendMessage(assistantMsg)

    try {
      const result = await runResearch({
        query: params.query,
        fast_model: params.fastModel,
        smart_model: params.smartModel,
        strategic_model: params.strategicModel,
        depth: params.depth,
        breadth: params.breadth,
        reasoning_effort: params.reasoningEffort === "none" ? null : params.reasoningEffort,
        report_type: params.reportType,
      })

      const sourcesList = result.sources.length
        ? `\n\n---\n**Sources (${result.sources.length})** | **Cost**: $${result.costs.toFixed(4)}\n${result.sources.map((s: string) => `- ${s}`).join("\n")}`
        : ""

      assistantMsg.content = result.report + sourcesList
      updateLast(assistantMsg)
    } catch (e) {
      setError((e as Error).message || "Research failed")
      assistantMsg.content = `Research error: ${(e as Error).message}`
      updateLast(assistantMsg)
    }
  }, [])

  return { research, error }
}