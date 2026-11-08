import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { DownloadPdfButton } from "../DownloadPdfButton"

const PDF_URL = "http://backend.test/fileviewer/raw?path=%2Fp%2F.latex-build%2Fslides.view.pdf&v=3"

/** The blobs saved, in order, with the name each was saved under. */
let saved: { name: string; href: string }[]
const fetchPdf = vi.fn<(url: string) => Promise<Response>>()

beforeEach(() => {
  saved = []
  vi.stubGlobal("fetch", fetchPdf)
  vi.stubGlobal("URL", class extends URL {
    static createObjectURL = () => "blob:pdf"
    static revokeObjectURL = () => {}
  })
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) {
    saved.push({ name: this.download, href: this.href })
  })
})

afterEach(() => {
  cleanup()
  fetchPdf.mockReset()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe("DownloadPdfButton", () => {
  it("fetches the PDF and saves it under the file name", async () => {
    fetchPdf.mockResolvedValue(new Response(new Blob(["%PDF"])))
    render(<DownloadPdfButton pdfUrl={PDF_URL} fileName="slides.pdf" />)

    fireEvent.click(screen.getByRole("button", { name: "Download PDF" }))

    await waitFor(() => expect(saved).toEqual([{ name: "slides.pdf", href: "blob:pdf" }]))
    expect(fetchPdf).toHaveBeenCalledExactlyOnceWith(PDF_URL)
  })

  it("is disabled without a PDF", () => {
    render(<DownloadPdfButton pdfUrl={null} fileName="slides.pdf" />)

    expect(screen.getByRole("button", { name: "Download PDF" })).toBeDisabled()
  })

  it("saves nothing when the PDF could not be fetched", async () => {
    fetchPdf.mockResolvedValue(new Response("gone", { status: 404 }))
    render(<DownloadPdfButton pdfUrl={PDF_URL} fileName="slides.pdf" />)

    fireEvent.click(screen.getByRole("button", { name: "Download PDF" }))

    await waitFor(() => expect(screen.getByRole("button", { name: "Download PDF" })).toHaveClass("text-danger"))
    expect(saved).toEqual([])
  })
})
