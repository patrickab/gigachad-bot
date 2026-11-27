"use client"

import { useEffect, useState } from "react"
import { useDebouncedSearch } from "@/hooks/useDebouncedSearch"
import { listAllChats, searchChats, type ChatListing, type ChatSearchHit } from "@/lib/backend/api"
import { errorMessage } from "@/lib/utils"

/** Longest query the backend searches for. */
export const QUERY_MAX_LENGTH = 200

/** Every chat of the login, read on mount; null until the read answers. */
export function useAllChats(): { chats: ChatListing[] | null; error: string } {
  const [state, setState] = useState<{ chats: ChatListing[] | null; error: string }>({ chats: null, error: "" })
  useEffect(() => {
    const controller = new AbortController()
    listAllChats(controller.signal).then(
      (chats) => { if (!controller.signal.aborted) setState({ chats, error: "" }) },
      (cause: unknown) => { if (!controller.signal.aborted) setState({ chats: [], error: errorMessage(cause, "Could not list chats") }) },
    )
    return () => controller.abort()
  }, [])
  return state
}

const searchChatHits = (query: string, signal: AbortSignal) => searchChats(query, 20, signal)

/** Content hits of `query`, read once typing pauses; a superseded request is aborted. `hits` is null while they are read. */
export function useChatSearch(query: string): { hits: ChatSearchHit[] | null; error: string } {
  return useDebouncedSearch(query.trim().slice(0, QUERY_MAX_LENGTH), searchChatHits)
}
