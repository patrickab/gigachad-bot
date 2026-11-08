"use client"

import { useEffect, useState } from "react"
import type { PdfDocument } from "@/components/content/pdf/pdfLayout"
import { cn } from "@/lib/utils"
import { slideNotes } from "./slideNotes"
import { SlideView } from "./SlideView"
import type { AudienceWindow } from "./useAudienceWindow"

const AUDIENCE_BUTTON_CLASS = "rounded px-1 font-sans text-ink-muted hover:bg-hover hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted"

/** The speaker notes of page `pageNumber`, empty while they load or when it has none. */
function usePageNotes(pdf: PdfDocument, pageNumber: number): string {
  const [notes, setNotes] = useState<{ pageNumber: number; text: string } | null>(null)
  useEffect(() => {
    let current = true
    pdf.getPage(pageNumber)
      .then((page) => page.getAnnotations())
      .then((annotations) => slideNotes(annotations), () => "")
      .then((text) => {
        if (current) setNotes({ pageNumber, text })
      })
    return () => {
      current = false
    }
  }, [pdf, pageNumber])
  return notes?.pageNumber === pageNumber ? notes.text : ""
}

/** `m:ss`, or `h:mm:ss` from one hour on. */
function formatElapsed(totalSeconds: number): string {
  const hours = Math.floor(totalSeconds / 3600)
  const minutes = Math.floor(totalSeconds / 60) % 60
  const seconds = String(totalSeconds % 60).padStart(2, "0")
  return hours > 0 ? `${hours}:${String(minutes).padStart(2, "0")}:${seconds}` : `${minutes}:${seconds}`
}

/** Time since it mounted, remount it to restart from zero. */
function ElapsedTimer() {
  const [seconds, setSeconds] = useState(0)
  useEffect(() => {
    const start = Date.now()
    const timer = setInterval(() => setSeconds(Math.floor((Date.now() - start) / 1000)), 1000)
    return () => clearInterval(timer)
  }, [])
  return <span>{formatElapsed(seconds)}</span>
}

/** The presenter's screen: the current slide large, the next one, or a muted "End" after the last slide, small above the current slide's notes, the slide count, the time since the last timer reset and, until it is open, a button for the audience window. */
export function PresenterView({ pdf, aspects, page, timerResets, audience }: { pdf: PdfDocument; aspects: number[]; page: number; timerResets: number; audience: AudienceWindow }) {
  const notes = usePageNotes(pdf, page)
  const next = page < pdf.numPages ? page + 1 : null

  return <div className="flex h-full flex-col gap-4 p-6">
    <div className="flex min-h-0 flex-1 gap-6">
      <section aria-label="Current slide" className="min-w-0 flex-[3]">
        <SlideView pageNumber={page} aspects={aspects} />
      </section>
      <div className="flex min-w-0 flex-[2] flex-col gap-6">
        <section aria-label="Next slide" className="h-2/5 shrink-0 opacity-80">
          {next !== null
            ? <SlideView pageNumber={next} aspects={aspects} />
            : <p className="flex h-full items-center justify-center text-sm text-ink-faint">End</p>}
        </section>
        <section aria-label="Notes" className="min-h-0 flex-1 overflow-y-auto whitespace-pre-wrap text-lg leading-relaxed text-ink">
          {notes}
        </section>
      </div>
    </div>
    <footer className="flex items-center justify-between font-mono text-sm tabular-nums text-ink-muted">
      <span>{`${page} / ${pdf.numPages}`}</span>
      {audience.status !== "open" && <button
        type="button"
        onClick={audience.open}
        className={cn(AUDIENCE_BUTTON_CLASS, audience.status === "blocked" && "text-danger hover:text-danger")}
      >
        Open audience window
      </button>}
      <ElapsedTimer key={timerResets} />
    </footer>
  </div>
}
