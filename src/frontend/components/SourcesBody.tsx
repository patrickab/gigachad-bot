"use client"

import { Globe, Image as ImageIcon } from "lucide-react"
import type { WebSearchResult } from "@/lib/types"

/** A search result's sources without repeats by URL, in first-seen order. */
export function uniqueSources(result: WebSearchResult): WebSearchResult["sources"] {
  const seen = new Set<string>()
  return result.sources.filter((s) => !seen.has(s.url) && seen.add(s.url))
}

/** Context-panel body listing a web search's videos, images and source cards. */
export function SourcesBody({ result }: { result: WebSearchResult }) {
  const sources = uniqueSources(result)
  const images = [...new Set(result.images)]
  const videos = result.videos ?? []

  if (sources.length === 0 && images.length === 0 && videos.length === 0) {
    return <div className="flex items-center justify-center py-6 text-xs text-ink-faint">No sources</div>
  }

  return (
    <div className="p-2 space-y-2">
      {videos.length > 0 && (
        <div className="mb-2">
          <div className="flex items-center gap-1.5 mb-1.5">
            <Globe className="h-3 w-3 text-ink-subtle" />
            <span className="text-[9px] font-medium text-ink-subtle uppercase tracking-wider">Videos</span>
          </div>
          <div className="grid grid-cols-2 gap-1.5">
            {videos.slice(0, 4).map((v, i) => (
              <a key={i} href={v.url} target="_blank" rel="noopener noreferrer">
                <img
                  src={v.thumbnail}
                  alt=""
                  className="w-full h-16 object-cover rounded border border-divider hover:border-ink-muted transition-colors"
                  onError={(e) => { (e.target as HTMLImageElement).style.display = "none" }}
                />
              </a>
            ))}
          </div>
        </div>
      )}
      {images.length > 0 && (
        <div className="mb-2">
          <div className="flex items-center gap-1.5 mb-1.5">
            <ImageIcon className="h-3 w-3 text-ink-subtle" />
            <span className="text-[9px] font-medium text-ink-subtle uppercase tracking-wider">Images</span>
          </div>
          <div className="grid grid-cols-2 gap-1.5">
            {images.slice(0, 4).map((img, i) => (
              <a key={i} href={img} target="_blank" rel="noopener noreferrer">
                <img
                  src={img}
                  alt=""
                  className="w-full h-16 object-cover rounded border border-divider hover:border-ink-muted transition-colors"
                  onError={(e) => {
                    ;(e.target as HTMLImageElement).style.display = "none"
                  }}
                />
              </a>
            ))}
          </div>
        </div>
      )}
      {sources.map((s, i) => {
        const domain = s.url.replace(/^https?:\/\/(www\.)?/, "").split("/")[0]
        return (
          <a
            key={i}
            href={s.url}
            target="_blank"
            rel="noopener noreferrer"
            className="block rounded-md border border-divider bg-surface/40 p-2.5 hover:border-divider-strong hover:bg-surface/70 transition-colors"
          >
            <div className="flex items-center gap-1.5 mb-0.5">
              <Globe className="h-2.5 w-2.5 text-ink shrink-0" />
              <span className="text-[10px] font-medium text-ink truncate">{domain}</span>
            </div>
            {s.title && <div className="text-[10px] font-medium text-ink mb-0.5 line-clamp-2">{s.title}</div>}
            {s.content && <div className="text-[9px] text-ink-subtle line-clamp-3">{s.content.slice(0, 150)}</div>}
          </a>
        )
      })}
    </div>
  )
}
