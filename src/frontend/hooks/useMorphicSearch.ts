"use client"

import { useCallback, useRef, useState } from "react"
import { morphicFetch, parseMorphicStream } from "@/lib/morphic"
import type { Message, MorphicSearchParams } from "@/lib/types"

export interface UseMorphicSearchReturn {
  morphicSearch: (params: MorphicSearchParams, appendMessage: (msg: Message) => void, updateLast: (msg: Message) => void) => Promise<void>
  error: string | null
}

export function useMorphicSearch(): UseMorphicSearchReturn {
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<(() => void) | null>(null)

  const morphicSearch = useCallback(async (
    params: MorphicSearchParams,
    appendMessage: (msg: Message) => void,
    updateLast: (msg: Message) => void,
  ) => {
    setError(null)

    const userMsg: Message = { role: "user", content: params.query }
    const assistantMsg: Message = { role: "assistant", content: "" }

    appendMessage(userMsg)
    appendMessage(assistantMsg)

    try {
      const { promise, abort } = morphicFetch(params)
      abortRef.current = abort

      const acc = { text: "", sources: [] as { title: string; url: string; content: string }[], images: [] as string[], query: params.query, citationMap: undefined as Record<string, { title: string; url: string; content: string }> | undefined }

      const res = await promise

      for await (const event of parseMorphicStream(res)) {
        if (event.type === "text" && event.text) {
          acc.text += event.text
          assistantMsg.content = acc.text
          updateLast(assistantMsg)
        } else if (event.type === "source") {
          if (event.sources) acc.sources = [...acc.sources, ...event.sources]
          if (event.images) acc.images = [...new Set([...acc.images, ...event.images])]
          if (event.query) acc.query = event.query
          if (event.citationMap) acc.citationMap = event.citationMap
          assistantMsg.morphic_result = { query: acc.query, sources: acc.sources, images: acc.images, citationMap: acc.citationMap }
          updateLast(assistantMsg)
        } else if (event.type === "error") {
          throw new Error(event.text || "Morphic search error")
        }
      }
    } catch (e) {
      if ((e as Error).name === "AbortError") return
      const msg = (e as Error)?.message ?? "Search failed"
      setError(msg)
      assistantMsg.content = assistantMsg.content || `Search error: ${msg}`
      updateLast(assistantMsg)
    } finally {
      abortRef.current = null
    }
  }, [])

  return { morphicSearch, error }
}