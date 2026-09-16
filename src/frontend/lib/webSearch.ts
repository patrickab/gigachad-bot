import type { WebSearchParams } from "./types"
import { getApiBase } from "./config"
import { readLines } from "./sse"

export interface WebSearchResultItem {
  label?: string
  title: string
  url: string
  content: string
}

/** Retained to render media from existing saved chats. New Brave searches do not fetch media. */
export interface WebSearchVideo {
  title: string
  url: string
  thumbnail: string
  iframe?: string
}

export interface WebSearchParsedEvent {
  type: "text" | "sources" | "error" | "done"
  text?: string
  sources?: WebSearchResultItem[]
}

/** Prepend the domain-filter tokens (e.g. "site:nature.com -reddit.com") to the query. */
export function applyDomainFilter(query: string, domain: string): string {
  const d = domain.trim()
  return d ? `${d} ${query}` : query
}

export function webSearchFetch(params: WebSearchParams) {
  const controller = new AbortController()
  const promise = fetch(`${getApiBase()}/web-search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: applyDomainFilter(params.query, params.domain ?? ""),
      system_instructions: params.systemInstructions ?? "",
      model: params.model ?? "",
    }),
    signal: controller.signal,
  })
  return { promise, abort: () => controller.abort() }
}

export async function* parseWebSearchStream(res: Response): AsyncGenerator<WebSearchParsedEvent> {
  if (!res.ok || !res.body) {
    yield { type: "error", text: await res.text() }
    return
  }
  try {
    for await (const line of readLines(res)) {
      let raw = line.trim()
      if (!raw) continue
      if (raw.startsWith("data: ")) raw = raw.slice(6)
      else if (raw.startsWith("data:")) raw = raw.slice(5)
      try {
        const event = JSON.parse(raw)
        if (event.type === "text" && typeof event.text === "string") yield { type: "text", text: event.text }
        else if (event.type === "sources" && Array.isArray(event.sources)) yield { type: "sources", sources: event.sources }
        else if (event.type === "done") yield { type: "done" }
        else if (event.type === "error") yield { type: "error", text: event.data ?? "Search failed" }
      } catch { /* skip keepalives */ }
    }
  } catch (e) {
    if (e instanceof Error && e.name === "AbortError") return
    yield { type: "error", text: e instanceof Error ? e.message : String(e) }
  }
}
