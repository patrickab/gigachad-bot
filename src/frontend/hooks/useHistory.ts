"use client"

import { useCallback, useState } from "react"
import { listChatHistories, loadChatHistory as loadApiHistory } from "@/lib/api"
import type { Message } from "@/lib/types"

export interface UseHistoryReturn {
  histories: Record<string, string[]>
  historiesLoading: boolean
  refreshHistories: () => Promise<void>
  loadHistory: (filename: string) => Promise<{ messages: Message[]; chat_id: string | null }>
  error: string | null
}

export function useHistory(): UseHistoryReturn {
  const [histories, setHistories] = useState<Record<string, string[]>>({})
  const [historiesLoading, setHistoriesLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const refreshHistories = useCallback(async () => {
    try {
      const data = await listChatHistories()
      setHistories(data.histories ?? {})
    } catch {
      // unavailable
    } finally {
      setHistoriesLoading(false)
    }
  }, [])

  const loadHistory = useCallback(async (filename: string): Promise<{ messages: Message[]; chat_id: string | null }> => {
    try {
      const data = await loadApiHistory(filename)
      return { messages: data.messages ?? [], chat_id: data.chat_id ?? null }
    } catch (e) {
      setError((e as Error).message)
      return { messages: [], chat_id: null }
    }
  }, [])

  return { histories, historiesLoading, refreshHistories, loadHistory, error }
}