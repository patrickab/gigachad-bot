"use client"

import { memo, useMemo, useState } from "react"
import { ChevronRight, Globe, PanelRightClose, ImageIcon } from "lucide-react"
import type { MorphicSearchResult } from "@/lib/types"

interface Props {
  result?: MorphicSearchResult
}

function SourcesSidebarInner({ result }: Props) {
  const [open, setOpen] = useState(true)

  const { sources, images } = useMemo(() => {
    if (!result) return { sources: [], images: [] }
    const seen = new Set<string>()
    const sources = result.sources.filter(s => !seen.has(s.url) && seen.add(s.url))
    const images = [...new Set(result.images)]
    return { sources, images }
  }, [result])

  if (!result || (sources.length === 0 && images.length === 0)) return null

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="absolute right-0 top-0 z-20 h-full w-6 flex items-center justify-center bg-zinc-900/80 border-l border-zinc-800 hover:bg-zinc-800 transition-colors"
        title="Show sources"
      >
        <Globe className="h-3 w-3 text-sky-400" />
      </button>
    )
  }

  return (
    <div className="shrink-0 w-64 border-l border-zinc-800 bg-zinc-950 flex flex-col overflow-hidden">
      <div className="flex items-center gap-2 px-3 py-2 border-b border-zinc-800/50">
        <Globe className="h-3.5 w-3.5 text-sky-400 shrink-0" />
        <span className="text-[11px] font-medium text-zinc-400 truncate flex-1">{result.query}</span>
        <span className="text-[10px] text-zinc-600 shrink-0">{sources.length}</span>
        <button onClick={() => setOpen(false)} className="rounded p-0.5 text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800 transition-colors" title="Hide sources">
          <PanelRightClose className="h-3.5 w-3.5" />
        </button>
      </div>
      <div className="flex-1 overflow-y-auto p-2 space-y-2">
        {images.length > 0 && (
          <div className="mb-2">
            <div className="flex items-center gap-1.5 mb-1.5">
              <ImageIcon className="h-3 w-3 text-zinc-500" />
              <span className="text-[9px] font-medium text-zinc-500 uppercase tracking-wider">Images</span>
            </div>
            <div className="grid grid-cols-2 gap-1.5">
              {images.slice(0, 4).map((img, i) => (
                <a key={i} href={img} target="_blank" rel="noopener noreferrer">
                  <img src={img} alt="" className="w-full h-16 object-cover rounded border border-zinc-800 hover:border-zinc-600 transition-colors" onError={e => { (e.target as HTMLImageElement).style.display = "none" }} />
                </a>
              ))}
            </div>
          </div>
        )}
        {sources.map((s, i) => {
          const domain = s.url.replace(/^https?:\/\/(www\.)?/, "").split("/")[0]
          return (
            <a key={i} href={s.url} target="_blank" rel="noopener noreferrer" className="block rounded-md border border-zinc-800 bg-zinc-900/40 p-2.5 hover:border-zinc-700 hover:bg-zinc-900/70 transition-colors">
              <div className="flex items-center gap-1.5 mb-0.5">
                <Globe className="h-2.5 w-2.5 text-sky-400 shrink-0" />
                <span className="text-[10px] font-medium text-sky-400 truncate">{domain}</span>
              </div>
              {s.title && <div className="text-[10px] font-medium text-zinc-300 mb-0.5 line-clamp-2">{s.title}</div>}
              {s.content && <div className="text-[9px] text-zinc-500 line-clamp-3">{s.content.slice(0, 150)}</div>}
            </a>
          )
        })}
      </div>
    </div>
  )
}

export const SourcesSidebar = memo(SourcesSidebarInner)