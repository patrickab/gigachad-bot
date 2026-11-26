"use client"

import { useEffect, useState } from "react"
import { listAllChats, searchChats, type ChatListing, type ChatSearchHit } from "@/lib/backend/api"
import { errorMessage } from "@/lib/utils"

/** Typing pauses this long before the chats are searched. */
export const SEARCH_DELAY_MS = 200
/** Longest query the backend searches for. */
const QUERY_MAX_LENGTH = 200

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

/** Content hits of `query`, read once typing pauses; a superseded request is aborted. `hits` is null while they are read. */
export function useChatSearch(query: string): { hits: ChatSearchHit[] | null; error: string } {
  const [found, setFound] = useState<{ query: string; hits: ChatSearchHit[]; error: string } | null>(null)
  const trimmed = query.trim().slice(0, QUERY_MAX_LENGTH)
  useEffect(() => {
    if (!trimmed) return
    const controller = new AbortController()
    const timer = setTimeout(() => {
      searchChats(trimmed, 20, controller.signal).then(
        (hits) => { if (!controller.signal.aborted) setFound({ query: trimmed, hits, error: "" }) },
        (cause: unknown) => { if (!controller.signal.aborted) setFound({ query: trimmed, hits: [], error: errorMessage(cause, "Search failed") }) },
      )
    }, SEARCH_DELAY_MS)
    return () => { clearTimeout(timer); controller.abort() }
  }, [trimmed])
  return found?.query === trimmed ? { hits: found.hits, error: found.error } : { hits: null, error: "" }
}
