import { useCallback, useState } from "react"

/** Characters of an answer a collapsed turn renders: a few times what its preview shows, so the preview can still scroll. */
export const PREVIEW_HEAD_CHARS = 1500

const FENCE = /^\s{0,3}(```|~~~)/

/**
 * The head of `markdown` for a collapsed turn's preview: cut at the first blank line past `limit` characters that is
 * outside code fences and `$$` blocks, so the head parses like the start of the whole. Shorter markdown is returned as is.
 *
 * A block still open four limits in is cut at a line end.
 */
export function previewHead(markdown: string, limit: number = PREVIEW_HEAD_CHARS): string {
  if (markdown.length <= limit) return markdown
  let offset = 0
  let inFence = false
  let inMath = false
  for (const line of markdown.split("\n")) {
    if (FENCE.test(line)) inFence = !inFence
    else if (!inFence && (line.split("$$").length - 1) % 2 === 1) inMath = !inMath
    else if (offset >= limit && !inFence && !inMath && line.trim() === "") return markdown.slice(0, offset)
    if (offset >= limit * 4) return markdown.slice(0, offset)
    offset += line.length + 1
  }
  return markdown
}

/**
 * Whether a collapsed turn's preview shows the whole answer: from the reader's first scroll or focus in it on.
 * `attach` is the ref of an element inside `TurnPreview`'s scroll region.
 */
export function usePreviewInteraction(): { full: boolean; attach: (host: HTMLElement | null) => (() => void) | undefined } {
  const [full, setFull] = useState(false)
  const attach = useCallback((host: HTMLElement | null) => {
    const region = host?.closest(".gc-turn-preview-scroll")
    if (!region) return
    const show = () => setFull(true)
    region.addEventListener("scroll", show, { passive: true })
    region.addEventListener("focusin", show)
    return () => {
      region.removeEventListener("scroll", show)
      region.removeEventListener("focusin", show)
    }
  }, [])
  return { full, attach }
}
