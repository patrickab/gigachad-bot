"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { createChatStream, fetchModels, fetchPrompts, listChatHistories, loadChatHistory as loadApiHistory, runResearch } from "@/lib/api"
import { morphicFetch, parseMorphicStream } from "@/lib/morphic"
import type { ChatRequest, Message, ModelsResponse, MorphicSearchParams } from "@/lib/types"

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

export interface UseChatReturn {
  messages: Message[]
  isStreaming: boolean
  send: (req: ChatRequest) => Promise<void>
  cancel: () => void
  reset: () => Promise<void>
  research: (params: ResearchParams) => Promise<void>
  morphicSearch: (params: MorphicSearchParams) => Promise<void>
  models: ModelsResponse | null
  prompts: string[]
  histories: Record<string, string[]>
  historiesLoading: boolean
  refreshHistories: () => Promise<void>
  error: string | null
  loadHistory: (filename: string) => Promise<void>
  deleteMessagePair: (index: number) => void
  addMessagePair: (userContent: string, assistantContent: string) => void
}

export function useChat(): UseChatReturn {
  const [messages, setMessages] = useState<Message[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [models, setModels] = useState<ModelsResponse | null>(null)
  const [prompts, setPrompts] = useState<string[]>([])
  const [histories, setHistories] = useState<Record<string, string[]>>({})
  const [historiesLoading, setHistoriesLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<(() => void) | null>(null)
  const assistantRef = useRef<Message | null>(null)

  useEffect(() => {
    fetchModels().then(setModels).catch(console.error)
    fetchPrompts().then(setPrompts).catch(console.error)
    loadHistories()
  }, [])

  async function loadHistories() {
    try {
      const data = await listChatHistories()
      setHistories(data.histories ?? {})
    } catch {
      // unavailable
    } finally {
      setHistoriesLoading(false)
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
        const { stream, abort } = createChatStream({ ...req, messages: [] })
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

  const morphicSearch = useCallback(async (params: MorphicSearchParams) => {
    setError(null)
    setIsStreaming(true)

    const userMsg: Message = { role: "user", content: params.query }
    const assistantMsg: Message = { role: "assistant", content: "" }
    assistantRef.current = assistantMsg

    setMessages((prev) => [...prev, userMsg, assistantMsg])

    try {
      const { promise, abort } = morphicFetch(params)
      abortRef.current = abort

      const acc = { text: "", sources: [] as { title: string; url: string; content: string }[], images: [] as string[], query: params.query, citationMap: undefined as Record<string, { title: string; url: string; content: string }> | undefined }

      const res = await promise
      const set = () => setMessages((prev) => { const c = [...prev]; c[c.length - 1] = { ...assistantMsg }; return c })

      for await (const event of parseMorphicStream(res)) {
        if (event.type === "text" && event.text) {
          acc.text += event.text
          assistantMsg.content = acc.text
          set()
        } else if (event.type === "source") {
          if (event.sources) acc.sources = [...acc.sources, ...event.sources]
          if (event.images) acc.images = [...new Set([...acc.images, ...event.images])]
          if (event.query) acc.query = event.query
          if (event.citationMap) acc.citationMap = event.citationMap
          assistantMsg.morphic_result = { query: acc.query, sources: acc.sources, images: acc.images, citationMap: acc.citationMap }
          set()
        } else if (event.type === "error") {
          throw new Error(event.text || "Morphic search error")
        }
      }
    } catch (e) {
      if ((e as Error).name === "AbortError") return
      const msg = (e as Error)?.message ?? "Search failed"
      setError(msg)
      assistantMsg.content = assistantMsg.content || `Search error: ${msg}`
      setMessages((prev) => { const c = [...prev]; c[c.length - 1] = { ...assistantMsg }; return c })
    } finally {
      setIsStreaming(false)
      abortRef.current = null
      assistantRef.current = null
    }
  }, [])

  const addMessagePair = useCallback((userContent: string, assistantContent: string) => {
    setMessages((prev) => [
      ...prev,
      { role: "user", content: userContent },
      { role: "assistant", content: assistantContent },
    ])
  }, [])

  return {
    messages,
    isStreaming,
    send,
    cancel,
    research,
    morphicSearch,
    reset,
    models,
    prompts,
    histories,
    historiesLoading,
    refreshHistories: loadHistories,
    error,
    loadHistory: loadHistoryMessages,
    deleteMessagePair,
    addMessagePair,
  }
}