"use client"

import { memo, useMemo } from "react"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"
import type { Message } from "@/lib/domain/types"

/** Cut the answer at raw JSON UI blocks and at trailing sections the UI renders separately (references, related questions). */
function truncateAtSections(text: string): string {
  // Safety net for generative UI ops leaking into the text
  const jsonIdx = text.indexOf('{"op":"add"')
  if (jsonIdx !== -1) text = text.slice(0, jsonIdx)

  // Match a heading of any level whose title is a known trailing section
  const headingRegex = /\n#{1,6}\s+(?:Related\s*Questions|References|Sources|Citations|Images|IMAGES|Further\s*Questions?|See\s*Also|Suggested\s*Questions?)\b/i
  const idx = text.search(headingRegex)
  if (idx !== -1) text = text.slice(0, idx)

  return text.trim()
}

interface Props {
  content: string
  search_result: NonNullable<Message["search_result"]>
}

/** Answer text with image markup and trailing sections stripped and citation markers rewritten to links for `citationMap`. */
function WebSearchResultInner({ content, search_result }: Props) {
  const text = useMemo(() => {
    let text = content

    // Strip image placeholders and markdown
    text = text.replace(/\s*\[Image\s*\d+\]\s*/gi, " ")
    text = text.replace(/!\[[^\]]*\]\([^)]*\)/g, "")
    text = text.replace(/!\[[^\]]*\]\[[^\]]*\]/g, "")
    text = text.replace(/<img[^>]*>/gi, "")

    // Truncate everything from the first section heading
    text = truncateAtSections(text)

    // Resolve citation markers to URLs using citationMap
    const cm = search_result.citationMap
    if (cm && Object.keys(cm).length) {
      text = text
        // Brave source labels, and legacy numbered citations, become citation pills.
        .replace(/\[([a-z0-9][a-z0-9-]*)\](?!\()/gi, (match, label: string) => {
          const url = cm[label]?.url
          return url ? `[${label}](${url})` : match
        })
        // Legacy format: [N](#anchor) citations
        .replace(/\[(\d+)\]\(#[^)]*\)/g, (_, n: string) => {
          const url = cm[n]?.url
          return url ? `[${n}](${url})` : `[${n}]`
        })
        // CJK bracket format: 【N†...】
        .replace(/【(\d+)†[^】]*】/g, (_, n: string) => {
          const url = cm[n]?.url
          return url ? `[${n}](${url})` : `[${n}]`
        })
    }

    return text
  }, [content, search_result])

  return <LaTeXMarkdown content={text} citationMap={search_result.citationMap} />
}

export const WebSearchResult = memo(WebSearchResultInner)
