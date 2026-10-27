/**
 * The PDF loads alongside its annotation key, not after it.
 *
 * The viewer used to render nothing until `/fileviewer/annotations` answered, so every open
 * paid that round trip (and its CORS preflight) before the PDF download even started. Ink must
 * still join the one viewer that is already loading, a remount would load the PDF twice.
 */
import { useEffect, type ReactNode } from "react"
import { describe, it, expect, vi, beforeEach } from "vitest"
import { render, act } from "@testing-library/react"
import type { CanvasDocument } from "@/lib/domain/canvasDocument"

const viewer = vi.hoisted(() => ({
  mounts: 0,
  props: null as null | { url: string; renderPageOverlay?: (page: number, pageWidth: number) => ReactNode },
}))
vi.mock("@/components/content/PdfViewer", () => ({
  PdfViewer: (props: { url: string; renderPageOverlay?: (page: number, pageWidth: number) => ReactNode }) => {
    viewer.props = props
    useEffect(() => { viewer.mounts += 1 }, [])
    return <div data-testid="pdf" />
  },
}))

const annotations = vi.hoisted(() => ({ resolve: null as null | ((key: string) => void), reject: null as null | ((error: Error) => void) }))
vi.mock("@/lib/backend/api", () => ({
  openPdfAnnotations: vi.fn(() => {
    const { promise, resolve, reject } = Promise.withResolvers<string>()
    annotations.resolve = resolve
    annotations.reject = reject
    return promise
  }),
}))

const stroke = { id: "s1", page: 1, points: [[1, 2, 0.5], [3, 4, 0.5]], color: "#000000", width: 2 }
vi.mock("@/features/editor/canvas/useCollaborativeCanvas", () => ({
  useCollaborativeCanvas: () => ({
    document: { version: 1, frames: [], strokes: [stroke], attachments: [], texts: [] } as unknown as CanvasDocument,
    replace: () => {},
    status: "synced",
  }),
}))

import { AnnotatedPdfViewer } from "@/features/editor/pdf/AnnotatedPdfViewer"

beforeEach(() => {
  viewer.mounts = 0
  viewer.props = null
})

describe("AnnotatedPdfViewer", () => {
  it("starts loading the PDF before the annotation key arrives and keeps that viewer once ink joins", async () => {
    render(<AnnotatedPdfViewer path="PDFs/paper.pdf" />)

    expect(viewer.mounts).toBe(1)
    expect(viewer.props?.url).toContain("/fileviewer/raw?path=PDFs%2Fpaper.pdf")
    expect(viewer.props?.renderPageOverlay).toBeUndefined()

    await act(async () => annotations.resolve?.("pdf-annotation/abc.canvas"))

    expect(viewer.mounts).toBe(1)
    const { container } = render(<>{viewer.props?.renderPageOverlay?.(1, 600)}</>)
    expect(container.querySelectorAll("svg.pdf-ink path")).toHaveLength(1)
  })

  it("stays a plain viewer when annotations are unavailable", async () => {
    render(<AnnotatedPdfViewer path="PDFs/paper.pdf" />)

    await act(async () => annotations.reject?.(new Error("offline")))

    expect(viewer.mounts).toBe(1)
    expect(viewer.props?.renderPageOverlay).toBeUndefined()
  })
})
