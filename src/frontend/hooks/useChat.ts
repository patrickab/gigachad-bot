"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { createChatStream, fetchHistory, fetchModels, fetchPrompts, listChatHistories, loadChatHistory as loadApiHistory, resetHistory, runResearch, runTavilySearch } from "@/lib/api"
import type { ChatRequest, Message, ModelsResponse } from "@/lib/types"

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

export interface WebSearchParams {
  query: string
  numQueries: number
  resultsPerQuery: number
}

export interface UseChatReturn {
  messages: Message[]
  isStreaming: boolean
  send: (req: ChatRequest) => Promise<void>
  cancel: () => void
  reset: () => Promise<void>
  research: (params: ResearchParams) => Promise<void>
  webSearch: (params: WebSearchParams) => Promise<void>
  models: ModelsResponse | null
  prompts: string[]
  histories: Record<string, string[]>
  refreshHistories: () => Promise<void>
  error: string | null
  loadHistory: (filename: string) => Promise<void>
  deleteMessagePair: (index: number) => void
}

export function useChat(): UseChatReturn {
  const [messages, setMessages] = useState<Message[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [models, setModels] = useState<ModelsResponse | null>(null)
  const [prompts, setPrompts] = useState<string[]>([])
  const [histories, setHistories] = useState<Record<string, string[]>>({})
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<(() => void) | null>(null)
  const assistantRef = useRef<Message | null>(null)

  useEffect(() => {
    fetchModels().then(setModels).catch(console.error)
    fetchPrompts().then(setPrompts).catch(console.error)
    loadInitialHistory()
    loadHistories()
  }, [])

  async function loadInitialHistory() {
    try {
      const data = await fetchHistory()
      setMessages(data.messages ?? [])
    } catch {
      // no saved state
    }
  }

  async function loadHistories() {
    try {
      const data = await listChatHistories()
      setHistories(data.histories ?? {})
    } catch {
      // unavailable
    }
  }

  const send = useCallback(
    async (req: ChatRequest) => {
      setError(null)
      setIsStreaming(true)

      const userMsg: Message = { role: "user", content: req.user_msg }
      const assistantMsg: Message = { role: "assistant", content: "" }
      assistantRef.current = assistantMsg

      setMessages((prev) => [...prev, userMsg, assistantMsg])

      try {
        const { stream, abort } = createChatStream(req)
        abortRef.current = abort

        const reader = stream
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          const text = new TextDecoder().decode(value, { stream: true })
          if (text) {
            assistantMsg.content += text
            setMessages((prev) => {
              const copy = [...prev]
              copy[copy.length - 1] = { ...assistantMsg }
              return copy
            })
          }
        }
      } catch (e) {
        if ((e as Error).name === "AbortError") return
        setError((e as Error).message || "Stream error")
      } finally {
        setIsStreaming(false)
        abortRef.current = null
        assistantRef.current = null
      }
    },
    []
  )

  const cancel = useCallback(() => {
    abortRef.current?.()
    setIsStreaming(false)
  }, [])

  const reset = useCallback(async () => {
    await resetHistory()
    setMessages([])
  }, [])

  const loadHistoryMessages = useCallback(async (filename: string) => {
    try {
      const data = await loadApiHistory(filename)
      setMessages(data.messages ?? [])
    } catch (e) {
      setError((e as Error).message)
    }
  }, [])

  const deleteMessagePair = useCallback((index: number) => {
    setMessages((prev) => {
      const copy = [...prev]
      copy.splice(index, 2)
      return copy
    })
  }, [])

  const research = useCallback(async (params: ResearchParams) => {
    setError(null)
    setIsStreaming(true)

    const userMsg: Message = { role: "user", content: params.query }
    const assistantMsg: Message = { role: "assistant", content: "" }
    assistantRef.current = assistantMsg

    setMessages((prev) => [...prev, userMsg, assistantMsg])

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
      setMessages((prev) => {
        const copy = [...prev]
        copy[copy.length - 1] = { ...assistantMsg }
        return copy
      })
    } catch (e) {
      setError((e as Error).message || "Research failed")
      assistantMsg.content = `Research error: ${(e as Error).message}`
      setMessages((prev) => {
        const copy = [...prev]
        copy[copy.length - 1] = { ...assistantMsg }
        return copy
      })
    } finally {
      setIsStreaming(false)
      abortRef.current = null
      assistantRef.current = null
    }
  }, [])

  const webSearch = useCallback(async (params: WebSearchParams) => {
    setError(null)
    setIsStreaming(true)

    const userMsg: Message = { role: "user", content: params.query }
    const assistantMsg: Message = { role: "assistant", content: "" }
    assistantRef.current = assistantMsg

    setMessages((prev) => [...prev, userMsg, assistantMsg])

    try {
      const result = await runTavilySearch({
        query: params.query,
        num_queries: params.numQueries,
        results_per_query: params.resultsPerQuery,
        expander_model: "ollama/gemma3:4b",
      })

      if (result.results.length === 0) {
        assistantMsg.content = "**Web Search** — no results found."
      } else {
        const lines = result.results.map(
          (r) => `- **[${r.title}](${r.url})** (score: ${r.score.toFixed(3)})\n  ${r.content.slice(0, 300)}${r.content.length > 300 ? "..." : ""}`
        )
        const queriesLine = result.queries.length
          ? `\n> Search queries: ${result.queries.map((q) => `"${q}"`).join(", ")}\n`
          : ""
        assistantMsg.content = `**Web Search Results** (${result.results.length} results)\n${queriesLine}\n${lines.join("\n\n")}`
      }

      setMessages((prev) => {
        const copy = [...prev]
        copy[copy.length - 1] = { ...assistantMsg }
        return copy
      })
    } catch (e) {
      setError((e as Error).message || "Web search failed")
      assistantMsg.content = `Web search error: ${(e as Error).message}`
      setMessages((prev) => {
        const copy = [...prev]
        copy[copy.length - 1] = { ...assistantMsg }
        return copy
      })
    } finally {
      setIsStreaming(false)
      abortRef.current = null
      assistantRef.current = null
    }
  }, [])

  return {
    messages,
    isStreaming,
    send,
    cancel,
    research,
    webSearch,
    reset,
    models,
    prompts,
    histories,
    refreshHistories: loadHistories,
    error,
    loadHistory: loadHistoryMessages,
    deleteMessagePair,
  }
}
