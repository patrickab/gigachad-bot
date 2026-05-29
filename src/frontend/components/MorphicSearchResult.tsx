"use client"

import { memo, useMemo } from "react"
import { LaTeXMarkdown } from "./LaTeXMarkdown"
import { ImageIcon } from "lucide-react"
import type { Message } from "@/lib/types"

interface Props {
  content: string
  morphic_result?: Message["morphic_result"]
}

function MorphicSearchResultInner({ content, morphic_result }: Props) {
  const { text, images } = useMemo(() => {
    const cm = morphic_result?.citationMap
    let text = content
    if (cm && Object.keys(cm).length) {
      text = text.replace(/\[(\d+)\]\(#[^)]*\)/g, (_, n: string) => {
        const url = cm[n]?.url
        return url ? `[${n}](${url})` : `[${n}]`
      })
      text = text.replace(/【(\d+)†[^】]*】/g, (_, n: string) => {
        const url = cm[n]?.url
        return url ? `[${n}](${url})` : `[${n}]`
      })
    }
    const images = morphic_result ? [...new Set(morphic_result.images)] : []
    return { text, images }
  }, [content, morphic_result])

  const body = <LaTeXMarkdown content={text} />

  if (!morphic_result || (images.length === 0 && !morphic_result.sources?.length)) return body

  return (
    <div className="space-y-3">
      {body}
      {images.length > 0 && (
        <div className="mt-3">
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
    </div>
  )
}

export const MorphicSearchResult = memo(MorphicSearchResultInner)