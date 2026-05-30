"use client"

import { useCallback, useState } from "react"
import { listChatHistories, loadChatHistory as loadApiHistory } from "@/lib/api"
import type { Message } from "@/lib/types"

export interface UseHistoryReturn {
  histories: Record<string, string[]>
  historiesLoading: boolean
  refreshHistories: () => Promise<void>
  loadHistory: (filename: string) => Promise<Message[]>
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

  const loadHistory = useCallback(async (filename: string): Promise<Message[]> => {
    try {
      const data = await loadApiHistory(filename)
      const msgs = data.messages ?? []
      return msgs
    } catch (e) {
      setError((e as Error).message)
      return []
    }
  }, [])

  return { histories, historiesLoading, refreshHistories, loadHistory, error }
}