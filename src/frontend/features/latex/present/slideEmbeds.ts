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

/** The embeds of page `pageNumber`, none without `pdf` or while they load. */
export function usePageEmbeds(pdf: PdfDocument | undefined, pageNumber: number): SlideEmbed[] {
  const [loaded, setLoaded] = useState<{ pageNumber: number; embeds: SlideEmbed[] } | null>(null)
  useEffect(() => {
    if (!pdf) return
    let current = true
    pdf.getPage(pageNumber)
      .then(async (page) => slideEmbeds(await page.getAnnotations(), page.view), () => [])
      .then((embeds) => {
        if (current) setLoaded({ pageNumber, embeds })
      })
    return () => {
      current = false
    }
  }, [pdf, pageNumber])
  return pdf && loaded?.pageNumber === pageNumber ? loaded.embeds : []
}

/** The key an embedded applet hands back to the deck with `postMessage({ type: "deck-key", key })`, null for any other message. */
export function deckKey(message: unknown): string | null {
  if (typeof message !== "object" || message === null) return null
  const { type, key } = message as { type?: unknown; key?: unknown }
  return type === "deck-key" && typeof key === "string" ? key : null
}
