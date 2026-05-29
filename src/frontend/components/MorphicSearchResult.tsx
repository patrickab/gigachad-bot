"use client"

import { memo, useState, useMemo } from "react"
import { LaTeXMarkdown } from "./LaTeXMarkdown"
import { ChevronDown, ChevronRight, Globe, ImageIcon } from "lucide-react"
import type { Message } from "@/lib/types"

interface Props {
  content: string
  morphic_result?: Message["morphic_result"]
}

function MorphicSearchResultInner({ content, morphic_result }: Props) {
  const [open, setOpen] = useState(true)

  const { text, sources, images } = useMemo(() => {
    const cm = morphic_result?.citationMap
    const text = (!cm || !Object.keys(cm).length)
      ? content
      : content.replace(/\[(\d+)\]\(#[^)]*\)/g, (_, n: string) => {
          const url = cm[n]?.url
          return url ? `[${n}](${url})` : `[${n}]`
        })
    const seen = new Set<string>()
    const sources = morphic_result?.sources.filter(s => !seen.has(s.url) && seen.add(s.url)) ?? []
    const images = morphic_result ? [...new Set(morphic_result.images)] : []
    return { text, sources, images }
  }, [content, morphic_result])

  if (!morphic_result || (sources.length === 0 && images.length === 0)) {
    return <LaTeXMarkdown content={text} />
  }

  return (
    <div className="space-y-3">
      <LaTeXMarkdown content={text} />
      <div className="mt-4 pt-3 border-t border-zinc-800/50">
        {images.length > 0 && (
          <div className="mb-3">
            <div className="flex items-center gap-2 mb-2">
              <ImageIcon className="h-3.5 w-3.5 text-zinc-500" />
              <span className="text-[10px] font-medium text-zinc-500 uppercase tracking-wider">Images</span>
            </div>
            <div className="grid grid-cols-3 gap-2">
              {images.slice(0, 6).map((img, i) => (
                <a key={i} href={img} target="_blank" rel="noopener noreferrer">
                  <img src={img} alt="" className="w-full h-24 object-cover rounded-lg border border-zinc-800 hover:border-zinc-600 transition-colors" onError={e => { (e.target as HTMLImageElement).style.display = "none" }} />
                </a>
              ))}
            </div>
          </div>
        )}
        <button onClick={() => setOpen(!open)} className="w-full flex items-center gap-2 px-3 py-2 text-left text-xs font-medium text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900 rounded-lg transition-colors">
          {open ? <ChevronDown className="h-3.5 w-3.5 shrink-0" /> : <ChevronRight className="h-3.5 w-3.5 shrink-0" />}
          <Globe className="h-3 w-3 shrink-0 text-sky-400" />
          <span className="truncate">{morphic_result.query}</span>
          <span className="ml-auto shrink-0 text-zinc-600">{sources.length} source{sources.length !== 1 ? "s" : ""}</span>
        </button>
        {open && (
          <div className="mt-2 grid grid-cols-1 gap-2 max-h-80 overflow-y-auto">
            {sources.map((s, i) => {
              const domain = s.url.replace(/^https?:\/\/(www\.)?/, "").split("/")[0]
              return (
                <a key={i} href={s.url} target="_blank" rel="noopener noreferrer" className="block rounded-lg border border-zinc-800 bg-zinc-900/50 p-3 hover:border-zinc-700 hover:bg-zinc-900 transition-colors">
                  <div className="flex items-center gap-2 mb-1">
                    <Globe className="h-3 w-3 text-sky-400 shrink-0" />
                    <span className="text-xs font-medium text-sky-400 truncate">{domain}</span>
                  </div>
                  {s.title && <div className="text-xs font-medium text-zinc-200 mb-1 line-clamp-2">{s.title}</div>}
                  {s.content && <div className="text-[11px] text-zinc-500 line-clamp-3">{s.content.slice(0, 200)}</div>}
                </a>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}

export const MorphicSearchResult = memo(MorphicSearchResultInner)