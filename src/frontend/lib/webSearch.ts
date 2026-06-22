import type { WebSearchParams } from "./types"
import { API_BASE } from "./config"
import { readLines } from "./sse"

export interface WebSearchResultItem {
  title: string
  url: string
  content: string
}

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
  const promise = fetch(`${API_BASE}/web-search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: applyDomainFilter(params.query, params.domain ?? ""),
      focus_mode: params.focusMode,
      optimization_mode: params.optimizationMode,
      system_instructions: params.systemInstructions ?? "",
      model: params.model ?? "",
    }),
    signal: controller.signal,
  })
  return { promise, abort: () => controller.abort() }
}

interface VaneSource {
  pageContent?: string
  content?: string
  title?: string
  url?: string
  metadata?: { title?: string; url?: string }
}

function normalizeSource(s: VaneSource): WebSearchResultItem {
  return {
    title: s.metadata?.title ?? s.title ?? "",
    url: s.metadata?.url ?? s.url ?? "",
    content: s.pageContent ?? s.content ?? "",
  }
}

export async function* parseWebSearchStream(res: Response): AsyncGenerator<WebSearchParsedEvent> {
  if (!res.ok || !res.body) {
    yield { type: "error", text: await res.text() }
    return
  }

  // Vane streams raw JSON lines (no SSE `data:` prefix in its native format).
  // The backend normalizes them to `data: {...}` lines. Track block state to
  // reconstruct text deltas from Vane's replace-whole-value patches.
  let textBlockId: string | null = null
  let lastTextLength = 0

  try {
    for await (const line of readLines(res)) {
      let raw = line.trim()
      if (!raw) continue
      // Strip SSE data: prefix if the backend added one
      if (raw.startsWith("data: ")) raw = raw.slice(6)
      else if (raw.startsWith("data:")) raw = raw.slice(5)
      if (raw === "[DONE]") { yield { type: "done" }; continue }
      try {
        const p = JSON.parse(raw)

        if (p.type === "block") {
          const b = p.block
          if (b?.type === "text") {
            textBlockId = b.id
            lastTextLength = 0
            const initial = typeof b.data === "string" ? b.data : ""
            if (initial) {
              lastTextLength = initial.length
              yield { type: "text", text: initial }
            }
          } else if (b?.type === "source" && Array.isArray(b.data)) {
            yield { type: "sources", sources: b.data.map(normalizeSource) }
          }
        } else if (p.type === "updateBlock" && p.blockId === textBlockId) {
          for (const patch of p.patch ?? []) {
            if (patch.path === "/data" && typeof patch.value === "string") {
              const delta = patch.value.slice(lastTextLength)
              lastTextLength = patch.value.length
              if (delta) yield { type: "text", text: delta }
            }
          }
        } else if (p.type === "messageEnd") {
          yield { type: "done" }
        } else if (p.type === "error") {
          yield { type: "error", text: p.data ?? p.message ?? "Unknown error" }
        }
      } catch { /* skip non-JSON keepalives */ }
    }
    yield { type: "done" }
  } catch (e) {
    if (e instanceof Error && e.name === "AbortError") return
    yield { type: "error", text: e instanceof Error ? e.message : String(e) }
  }
}

export async function fetchWebSearchImages(query: string, model: string): Promise<string[]> {
  try {
    const r = await fetch(`${API_BASE}/web-search/images`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, model }),
    })
    if (!r.ok) return []
    const d = await r.json()
    return (d.images ?? []).map((i: { img_src?: string; url?: string }) => i.img_src ?? i.url).filter(Boolean)
  } catch {
    return []
  }
}

export async function fetchWebSearchVideos(query: string, model: string): Promise<WebSearchVideo[]> {
  try {
    const r = await fetch(`${API_BASE}/web-search/videos`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, model }),
    })
    if (!r.ok) return []
    const d = await r.json()
    return (d.videos ?? []).map((v: { title?: string; url?: string; img_src?: string; iframe_src?: string }) => ({
      title: v.title ?? "",
      url: v.url ?? "",
      thumbnail: v.img_src ?? "",
      iframe: v.iframe_src,
    }))
  } catch {
    return []
  }
}
