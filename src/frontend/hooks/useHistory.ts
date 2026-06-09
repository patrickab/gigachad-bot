"use client"

import { useCallback, useState } from "react"
import { loadChatHistory as loadApiHistory } from "@/lib/api"
import type { Message } from "@/lib/types"

export interface UseHistoryReturn {
  loadHistory: (filename: string) => Promise<{ messages: Message[]; chat_id: string | null; parent_id: string | null; branch_message_idx: number | null }>
}

export function useHistory(): UseHistoryReturn {
  const loadHistory = useCallback(async (filename: string): Promise<{ messages: Message[]; chat_id: string | null; parent_id: string | null; branch_message_idx: number | null }> => {
    try {
      const data = await loadApiHistory(filename)
      return { messages: data.messages ?? [], chat_id: data.chat_id ?? null, parent_id: data.parent_id ?? null, branch_message_idx: data.branch_message_idx ?? null }
    } catch {
      return { messages: [], chat_id: null, parent_id: null, branch_message_idx: null }
    }
  }, [])

  return { loadHistory }
}