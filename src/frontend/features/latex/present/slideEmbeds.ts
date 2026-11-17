import { useEffect, useState } from "react"
import type { PdfDocument } from "@/components/content/pdf/pdfLayout"

/** A live frame over part of a slide: its address and its box in fractions of the page. */
export type SlideEmbed = { url: string; left: number; top: number; width: number; height: number }

/** The fields of a pdf.js annotation that place a link. */
interface LinkAnnotation {
  subtype?: string
  url?: string
  rect?: number[]
}

function marksEmbed(url: string): boolean {
  return URL.canParse(url) && new URL(url).searchParams.get("embed") === "1"
}

/** A page's embeds: its links whose address carries `embed=1`, each over the link's rectangle. `view` is the page box `[x0, y0, x1, y1]` in PDF points, y upwards. */
export function slideEmbeds(annotations: LinkAnnotation[], view: number[]): SlideEmbed[] {
  const [x0, y0, x1, y1] = view
  const pageWidth = x1 - x0
  const pageHeight = y1 - y0
  return annotations.flatMap(({ subtype, url, rect }) => {
    if (subtype !== "Link" || !url || !rect || !marksEmbed(url)) return []
    const [left, bottom, right, top] = rect
    return [{ url, left: (left - x0) / pageWidth, top: (y1 - top) / pageHeight, width: (right - left) / pageWidth, height: (top - bottom) / pageHeight }]
  })
}

/** A deck's embeds with the page each one sits on. */
export type DeckEmbed = SlideEmbed & { page: number }

/** One live frame of the deck. A link whose address repeats on consecutive pages, as on a slide's overlays, keeps one frame
 * across them: `pages` are those pages in order, and the frame tells its applet which of them shows as its step. */
export type DeckFrame = { key: string; url: string; pages: DeckEmbed[] }

/** The deck's frames: every embed starts a frame, unless the page before has a frame with the same address, which it joins. */
export function deckFrames(embeds: DeckEmbed[]): DeckFrame[] {
  const frames: DeckFrame[] = []
  for (const embed of [...embeds].sort((a, b) => a.page - b.page)) {
    const run = frames.find((frame) => frame.url === embed.url && frame.pages[frame.pages.length - 1].page === embed.page - 1)
    if (run) run.pages.push(embed)
    else frames.push({ key: `${embed.page} ${embed.url}`, url: embed.url, pages: [embed] })
  }
  return frames
}

/** Which of a frame's pages shows, its step, or -1 while its pages are off screen. */
export function frameStep(frame: DeckFrame, pageNumber: number): number {
  return frame.pages.findIndex((embed) => embed.page === pageNumber)
}

/** Every embed of the deck in page order, none without `pdf` or while they load. */
export function useDeckEmbeds(pdf: PdfDocument | undefined): DeckEmbed[] {
  const [loaded, setLoaded] = useState<{ pdf: PdfDocument; embeds: DeckEmbed[] } | null>(null)
  useEffect(() => {
    if (!pdf) return
    let current = true
    const pageEmbeds = async (pageNumber: number): Promise<DeckEmbed[]> => {
      const page = await pdf.getPage(pageNumber)
      return slideEmbeds(await page.getAnnotations(), page.view).map((embed) => ({ ...embed, page: pageNumber }))
    }
    Promise.all(Array.from({ length: pdf.numPages }, (_, index) => pageEmbeds(index + 1).catch(() => [])))
      .then((pages) => {
        if (current) setLoaded({ pdf, embeds: pages.flat() })
      })
    return () => {
      current = false
    }
  }, [pdf])
  return pdf && loaded?.pdf === pdf ? loaded.embeds : []
}

/** The key an embedded applet hands back to the deck with `postMessage({ type: "deck-key", key })`, null for any other message. */
export function deckKey(message: unknown): string | null {
  if (typeof message !== "object" || message === null) return null
  const { type, key } = message as { type?: unknown; key?: unknown }
  return type === "deck-key" && typeof key === "string" ? key : null
}
