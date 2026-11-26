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
/** With a composer (`data-composer`) on the page, for chips and history attachments alike, the card is this share of its width. */
const COMPOSER_WIDTH_RATIO = 0.62
/** Composer and card together take at most this share of the viewport height. */
const VIEW_HEIGHT_RATIO = 0.62
const MIN_CARD_HEIGHT_PX = 120
const MIN_CARD_WIDTH_PX = 120
const CARD_PADDING_PX = 8
const CARD_BORDER_PX = 1
const CARD_GAP_PX = 8

/**
 * Scrollable card `bottom` px above the viewport bottom.
 *
 * - The scrollbar gutter is always reserved, so content laid out at `contentWidth` never overflows sideways
 * - A PDF card is exactly one page tall (within the space above the anchor) and scrolls to the next pages
 */
function HoverCard({ attachment, text, bottom, left, width, reservedHeight, onMouseEnter, onMouseLeave }: {
  attachment: Attachment
  text?: string
  bottom: number
  left: number
  /** Width at which the card fits the anchor, shrunk for an image or PDF page that would be taller than the height budget. */
  width: number
  /** Height of the composer the card sits above, which counts against the `VIEW_HEIGHT_RATIO` budget. */
  reservedHeight: number
  onMouseEnter: () => void
  onMouseLeave: () => void
}) {
  const [fullContentWidth, setFullContentWidth] = useState(0)
  const [aspect, setAspect] = useState<number | null>(null)
  const measure = useCallback((element: HTMLDivElement | null) => {
    if (element) setFullContentWidth(element.clientWidth - 2 * CARD_PADDING_PX)
  }, [])
  const isPdf = attachment.mime === PDF_MIME
  const budgetHeight = Math.max(MIN_CARD_HEIGHT_PX, window.innerHeight * VIEW_HEIGHT_RATIO - reservedHeight)
  // Only the budget shrinks the width: a smaller room above the anchor scrolls the card instead, so the width stays the composer's share
  const availableHeight = Math.min(window.innerHeight - bottom - CARD_GAP_PX, budgetHeight)
  // Scrollbar gutter and padding around the content, constant for any card width
  const chrome = width - fullContentWidth
  const verticalChrome = 2 * (CARD_PADDING_PX + CARD_BORDER_PX)
  const fitWidth = aspect && fullContentWidth > 0 ? (budgetHeight - verticalChrome) / aspect + chrome : width
  const cardWidth = Math.max(MIN_CARD_WIDTH_PX, Math.min(width, fitWidth))
  const contentWidth = fullContentWidth > 0 ? cardWidth - chrome : 0
  const pageHeight = isPdf && aspect && contentWidth > 0 ? contentWidth * aspect + verticalChrome : null

  return (
    <div
      ref={measure}
      aria-hidden="true"
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      style={{
        position: "fixed", bottom, left: left + (width - cardWidth) / 2, width: cardWidth, padding: CARD_PADDING_PX, borderWidth: CARD_BORDER_PX,
        height: pageHeight ? Math.min(pageHeight, availableHeight) : undefined,
        maxHeight: pageHeight ? undefined : availableHeight,
      }}
      className="z-[80] overflow-y-auto overflow-x-hidden overscroll-contain rounded-xl border-divider/50 bg-surface shadow-[var(--shadow-xl)] [scrollbar-gutter:stable]"
    >
      {contentWidth > 0 && <CardBody attachment={attachment} text={text} width={contentWidth} onPageAspect={setAspect} />}
    </div>
  )
}

function CardBody({ attachment, text, width, onPageAspect }: { attachment: Attachment; text?: string; width: number; onPageAspect: (ratio: number) => void }) {
  if (text !== undefined) return <pre className="whitespace-pre-wrap break-words px-1 font-mono text-[11px] text-ink">{text}</pre>
  if (attachment.mime.startsWith("image/")) {
    // eslint-disable-next-line @next/next/no-img-element -- attachment URLs are blob/data/backend URLs, not optimizable by next/image under Tauri static export
    return <img src={attachment.url} alt="" onLoad={(event) => onPageAspect(event.currentTarget.naturalHeight / event.currentTarget.naturalWidth)} className="w-full rounded object-contain" />
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
export function useAttachmentHoverCard(attachment: Attachment, text?: string) {
  const [position, setPosition] = useState<{ bottom: number; left: number; width: number; reservedHeight: number } | null>(null)
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
      const composer = (anchor.closest("[data-composer]") ?? document.querySelector("[data-composer]"))?.getBoundingClientRect()
      const width = composer ? composer.width * COMPOSER_WIDTH_RATIO : CARD_WIDTH_PX
      const centered = rect.left + rect.width / 2 - width / 2
      const left = Math.min(Math.max(centered, CARD_GAP_PX), window.innerWidth - width - CARD_GAP_PX)
      setPosition({ bottom: window.innerHeight - rect.top + CARD_GAP_PX, left, width, reservedHeight: composer?.height ?? 0 })
    }, SHOW_DELAY_MS)
  }, [clearTimers])

  useEffect(() => clearTimers, [clearTimers])

  const card = position && createPortal(
    <HoverCard attachment={attachment} text={text} bottom={position.bottom} left={position.left} width={position.width} reservedHeight={position.reservedHeight} onMouseEnter={keepOpen} onMouseLeave={hideSoon} />,
    document.body,
  )

  return {
    anchorProps: { onMouseEnter: scheduleShow, onMouseLeave: hideSoon, onFocus: scheduleShow, onBlur: hideNow, onMouseDown: hideNow },
    card,
  }
}
