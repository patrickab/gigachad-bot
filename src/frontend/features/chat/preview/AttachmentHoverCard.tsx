"use client"

import { useCallback, useEffect, useRef, useState, type FocusEvent, type MouseEvent } from "react"
import { createPortal } from "react-dom"
import type { Attachment } from "@/lib/domain/types"
import { rewriteImages } from "@/lib/backend/urls"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"
import { LazyPdfPages } from "@/components/content/LazyPdfViewer"
import { PDF_MIME, renderableMarkdown } from "./AttachmentView"

const SHOW_DELAY_MS = 300
/** Grace period to move the pointer from the anchor onto the card. */
const HIDE_DELAY_MS = 150
const CARD_WIDTH_PX = 340
const CARD_PADDING_PX = 8
const CARD_BORDER_PX = 1
const CARD_GAP_PX = 8

/**
 * Scrollable card `bottom` px above the viewport bottom.
 *
 * - The scrollbar gutter is always reserved, so content laid out at `contentWidth` never overflows sideways
 * - A PDF card is exactly one page tall (within the space above the anchor) and scrolls to the next pages
 */
function HoverCard({ attachment, bottom, left, onMouseEnter, onMouseLeave }: {
  attachment: Attachment
  bottom: number
  left: number
  onMouseEnter: () => void
  onMouseLeave: () => void
}) {
  const [contentWidth, setContentWidth] = useState(0)
  const [pageAspect, setPageAspect] = useState<number | null>(null)
  const measure = useCallback((element: HTMLDivElement | null) => {
    if (element) setContentWidth(element.clientWidth - 2 * CARD_PADDING_PX)
  }, [])
  const isPdf = attachment.mime === PDF_MIME
  const availableHeight = window.innerHeight - bottom - CARD_GAP_PX
  const pageHeight = isPdf && pageAspect && contentWidth > 0 ? contentWidth * pageAspect + 2 * (CARD_PADDING_PX + CARD_BORDER_PX) : null

  return (
    <div
      ref={measure}
      aria-hidden="true"
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      style={{
        position: "fixed", bottom, left, width: CARD_WIDTH_PX, padding: CARD_PADDING_PX, borderWidth: CARD_BORDER_PX,
        height: pageHeight ? Math.min(pageHeight, availableHeight) : undefined,
        maxHeight: pageHeight ? undefined : Math.min(window.innerHeight * 0.6, availableHeight),
      }}
      className="z-[80] overflow-y-auto overflow-x-hidden overscroll-contain rounded-xl border-divider/50 bg-surface shadow-[var(--shadow-xl)] [scrollbar-gutter:stable]"
    >
      {contentWidth > 0 && <CardBody attachment={attachment} width={contentWidth} onPageAspect={setPageAspect} />}
    </div>
  )
}

function CardBody({ attachment, width, onPageAspect }: { attachment: Attachment; width: number; onPageAspect: (ratio: number) => void }) {
  if (attachment.mime.startsWith("image/")) {
    // eslint-disable-next-line @next/next/no-img-element -- attachment URLs are blob/data/backend URLs, not optimizable by next/image under Tauri static export
    return <img src={attachment.url} alt="" className="w-full rounded object-contain" />
  }
  if (attachment.mime === PDF_MIME) return <LazyPdfPages url={attachment.url} width={width} onFirstPageAspect={onPageAspect} />
  const markdown = renderableMarkdown(attachment)
  if (!markdown) return <p className="px-1 py-2 text-xs text-ink-subtle">No preview available</p>
  return <div className="px-1 text-[11px]"><LaTeXMarkdown content={rewriteImages(markdown, attachment.name)} compact /></div>
}

/**
 * Peek at an attachment: hovering or focusing the anchor shows a scrollable preview card centered above it after a short delay.
 *
 * - The card stays open while the pointer is on it, so it can be scrolled
 * - Spread `anchorProps` on the element that opens the full preview, render `card` anywhere
 */
export function useAttachmentHoverCard(attachment: Attachment) {
  const [position, setPosition] = useState<{ bottom: number; left: number } | null>(null)
  const showTimer = useRef<number | undefined>(undefined)
  const hideTimer = useRef<number | undefined>(undefined)

  const clearTimers = useCallback(() => {
    window.clearTimeout(showTimer.current)
    window.clearTimeout(hideTimer.current)
  }, [])

  const hideNow = useCallback(() => {
    clearTimers()
    setPosition(null)
  }, [clearTimers])

  const hideSoon = useCallback(() => {
    window.clearTimeout(showTimer.current)
    hideTimer.current = window.setTimeout(() => setPosition(null), HIDE_DELAY_MS)
  }, [])

  const keepOpen = useCallback(() => window.clearTimeout(hideTimer.current), [])

  const scheduleShow = useCallback((event: MouseEvent<HTMLElement> | FocusEvent<HTMLElement>) => {
    const anchor = event.currentTarget
    clearTimers()
    showTimer.current = window.setTimeout(() => {
      const rect = anchor.getBoundingClientRect()
      const centered = rect.left + rect.width / 2 - CARD_WIDTH_PX / 2
      const left = Math.min(Math.max(centered, CARD_GAP_PX), window.innerWidth - CARD_WIDTH_PX - CARD_GAP_PX)
      setPosition({ bottom: window.innerHeight - rect.top + CARD_GAP_PX, left })
    }, SHOW_DELAY_MS)
  }, [clearTimers])

  useEffect(() => clearTimers, [clearTimers])

  const card = position && createPortal(
    <HoverCard attachment={attachment} bottom={position.bottom} left={position.left} onMouseEnter={keepOpen} onMouseLeave={hideSoon} />,
    document.body,
  )

  return {
    anchorProps: { onMouseEnter: scheduleShow, onMouseLeave: hideSoon, onFocus: scheduleShow, onBlur: hideNow, onMouseDown: hideNow },
    card,
  }
}
