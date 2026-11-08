"use client"

import { useState } from "react"
import { Document } from "react-pdf"
import "@/components/content/pdf/pdfjsWorker"
import { readPageAspects, type PdfDocument } from "@/components/content/pdf/pdfLayout"
import { LoadingPlaceholder } from "@/components/ui/DocumentLoadState"
import { useEventListener } from "@/hooks/useEventListener"
import { cn } from "@/lib/utils"
import type { PresentParams } from "./presentParams"
import { PresenterView } from "./PresenterView"
import { FALLBACK_ASPECT, SlideView } from "./SlideView"
import { useAudienceWindow } from "./useAudienceWindow"
import { useSlidePosition } from "./useSlidePosition"

/** Fullscreen needs a user gesture, so this only runs from a key or click handler. */
function enterFullscreen() {
  if (!document.fullscreenElement) document.documentElement.requestFullscreen?.().catch(() => {})
}

function exitFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen().catch(() => {})
}

type Deck = { pdf: PdfDocument; aspects: number[] }

/** One presentation window: `audience` shows the slide alone on black, `presenter` adds the next slide, notes and a timer. The slide follows every window on `channel`. */
export function PresentationWindow({ pdfUrl, role, channel }: PresentParams) {
  const [deck, setDeck] = useState<Deck | null>(null)
  const [timerResets, setTimerResets] = useState(0)
  const pageCount = deck?.pdf.numPages ?? 0
  const { page, go } = useSlidePosition(channel, pageCount)
  const audience = useAudienceWindow({ pdfUrl, role: "audience", channel })

  const next = () => go(page + 1)
  const previous = () => go(page - 1)
  const toggleFullscreen = () => (document.fullscreenElement ? exitFullscreen() : enterFullscreen())
  const resetTimer = () => setTimerResets((count) => count + 1)
  const openAudience = () => audience.open()
  // Enter only enters fullscreen, it never changes the slide. A opens the audience window, the presenter's only
  const keyHandlers: Record<string, () => void> = {
    ArrowRight: next,
    ArrowDown: next,
    " ": next,
    PageDown: next,
    ArrowLeft: previous,
    ArrowUp: previous,
    PageUp: previous,
    Home: () => go(1),
    End: () => go(pageCount),
    Enter: enterFullscreen,
    f: toggleFullscreen,
    F: toggleFullscreen,
    Escape: exitFullscreen,
    r: resetTimer,
    R: resetTimer,
    ...(role === "presenter" && { a: openAudience, A: openAudience }),
  }

  useEventListener("window", "keydown", (event) => {
    if (event.altKey || event.ctrlKey || event.metaKey) return
    const handler = keyHandlers[event.key]
    if (!handler) return
    event.preventDefault()
    handler()
  })

  const load = async (pdf: PdfDocument) => setDeck({ pdf, aspects: await readPageAspects(pdf, FALLBACK_ASPECT) })

  return <Document
    file={pdfUrl}
    onLoadSuccess={load}
    loading={<LoadingPlaceholder size="fill" />}
    error={<div className="flex h-full items-center justify-center text-sm text-ink-faint">Could not load the PDF</div>}
    className={cn("h-dvh", role === "audience" ? "bg-black" : "bg-paper text-ink")}
  >
    {deck && (role === "audience"
      ? <main aria-label="Slide" className="h-full" onClick={enterFullscreen}>
        <SlideView pageNumber={page} aspects={deck.aspects} />
      </main>
      : <PresenterView pdf={deck.pdf} aspects={deck.aspects} page={page} timerResets={timerResets} audience={audience} />)}
  </Document>
}
