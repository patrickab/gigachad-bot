"use client"

import { Globe } from "lucide-react"
import type { WebSearchResult } from "@/lib/domain/types"

const MEDIA_LIMIT = 4

/** A search result's sources without repeats by URL, in first-seen order. */
function uniqueSources(result: WebSearchResult): WebSearchResult["sources"] {
  const seen = new Set<string>()
  return result.sources.filter((s) => !seen.has(s.url) && seen.add(s.url))
}

/** Remote thumbnail linking to `href`, hidden when it fails to load. */
function MediaLink({ href, src }: { href: string; src: string }) {
  return (
    <a href={href} target="_blank" rel="noopener noreferrer">
      {/* eslint-disable-next-line @next/next/no-img-element -- arbitrary remote search-result URLs, unsuitable for next/image under Tauri static export */}
      <img
        src={src}
        alt=""
        className="h-16 w-full rounded border border-divider/50 object-cover transition-colors hover:border-ink-muted"
        onError={(e) => { (e.target as HTMLImageElement).style.display = "none" }}
      />
    </a>
  )
}

/** Collapsed "N sources" row under a searched answer, opening to its videos, images and source cards. */
export function WebSources({ result }: { result: WebSearchResult }) {
  const sources = uniqueSources(result)
  const media = [
    ...(result.videos ?? []).slice(0, MEDIA_LIMIT).map((video) => ({ href: video.url, src: video.thumbnail })),
    ...[...new Set(result.images)].slice(0, MEDIA_LIMIT).map((image) => ({ href: image, src: image })),
  ]
  if (sources.length === 0 && media.length === 0) return null

  return (
    <details className="group/sources mt-3">
      <summary className="flex w-fit cursor-pointer list-none select-none items-center gap-1.5 text-[10px] text-ink-subtle transition-colors hover:text-ink">
        <Globe className="h-3 w-3" aria-hidden="true" />
        <span className="font-medium uppercase tracking-wide">{sources.length} {sources.length === 1 ? "source" : "sources"}</span>
      </summary>
      <div className="mt-2 space-y-2">
        {media.length > 0 && (
          <div className="grid grid-cols-4 gap-1.5">
            {media.map((item) => <MediaLink key={item.href + item.src} href={item.href} src={item.src} />)}
          </div>
        )}
        <div className="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
          {sources.map((source) => {
            const domain = source.url.replace(/^https?:\/\/(www\.)?/, "").split("/")[0]
            return (
              <a
                key={source.url}
                href={source.url}
                target="_blank"
                rel="noopener noreferrer"
                className="block rounded-md border border-divider/50 p-2 transition-colors hover:border-divider-strong hover:bg-hover"
              >
                <span className="block truncate text-[10px] text-ink-subtle">{domain}</span>
                {source.title && <span className="line-clamp-2 block text-[11px] font-medium text-ink">{source.title}</span>}
              </a>
            )
          })}
        </div>
      </div>
    </details>
  )
}
