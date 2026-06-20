import type { MorphicSearchParams } from "./types"
import { API_BASE } from "./config"
import { readLines } from "./sse"

export interface MorphicResultItem {
  title: string
  url: string
  content: string
}

export interface MorphicParsedEvent {
  type: "text" | "source" | "error" | "done"
  text?: string
  sources?: MorphicResultItem[]
  images?: string[]
  query?: string
  citationMap?: Record<string, MorphicResultItem>
}

export function morphicFetch(params: MorphicSearchParams) {
  const controller = new AbortController()
  const promise = fetch(`${API_BASE}/morphic-search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: params.query,
      search_depth: params.searchDepth,
      ...(params.model ? { model: params.model } : {}),
    }),
    signal: controller.signal,
  })
  return { promise, abort: () => controller.abort() }
}

export async function* parseMorphicStream(
  res: Response
): AsyncGenerator<MorphicParsedEvent> {
  if (!res.ok || !res.body) {
    yield { type: "error", text: await res.text() }
    return
  }
  try {
    for await (const line of readLines(res)) {
      const t = line.trim()
      if (!t.startsWith("data:")) continue
      const json = t.startsWith("data: ") ? t.slice(6) : t.slice(5)
      if (json === "[DONE]") { yield { type: "done" }; continue }
      try {
        const p = JSON.parse(json)
        if (p.type === "text-delta" && typeof p.delta === "string") {
          yield { type: "text", text: p.delta }
        } else if (p.type === "tool-output-available" && p.output?.state === "complete" && p.output.results && !p.preliminary) {
          const o = p.output
          yield {
            type: "source",
            sources: o.results.map((r: MorphicResultItem) => ({ title: r.title ?? "", url: r.url ?? "", content: r.content ?? "" })),
            images: o.images ?? [],
            query: o.query ?? "",
            citationMap: o.citationMap,
          }
        } else if (p.type === "text" && typeof p.text === "string") {
          yield { type: "text", text: p.text }
        } else if (p.type === "error") {
          yield { type: "error", text: p.errorText ?? "Unknown error" }
        }
      } catch { /* skip */ }
    }
    yield { type: "done" }
  } catch (e) {
    if (e instanceof Error && e.name === "AbortError") return
    yield { type: "error", text: e instanceof Error ? e.message : String(e) }
  }
}