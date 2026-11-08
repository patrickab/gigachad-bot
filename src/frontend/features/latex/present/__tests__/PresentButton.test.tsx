import { afterEach, describe, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { PresentButton } from "../PresentButton"
import { parsePresentQuery } from "../presentParams"

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe("PresentButton", () => {
  it("opens only the presenter window, on a channel of its own", () => {
    const open = vi.spyOn(window, "open").mockImplementation(() => ({ close: vi.fn() }) as unknown as Window)
    const pdfUrl = "http://127.0.0.1:8001/api/fileviewer/raw?path=talk%2F.latex-build%2Ftalk.view.pdf&v=3"
    render(<PresentButton pdfUrl={pdfUrl} />)

    fireEvent.click(screen.getByRole("button", { name: "Present" }))

    expect(open).toHaveBeenCalledTimes(1)
    const url = new URL(String(open.mock.calls[0][0]), window.location.origin)
    expect(url.pathname).toBe("/present")
    const params = parsePresentQuery(url.search)
    expect(params?.role).toBe("presenter")
    expect(params?.pdfUrl).toBe(pdfUrl)
    expect(params?.channel).toBeTruthy()
  })

  it("turns red when the browser blocks the popup", () => {
    vi.spyOn(window, "open").mockReturnValue(null)
    render(<PresentButton pdfUrl="/deck.pdf" />)

    const button = screen.getByRole("button", { name: "Present" })
    fireEvent.click(button)

    expect(button.className).toContain("text-danger")
  })

  it("is disabled without a PDF", () => {
    const open = vi.spyOn(window, "open")
    render(<PresentButton pdfUrl={null} />)

    const button = screen.getByRole<HTMLButtonElement>("button", { name: "Present" })
    expect(button.disabled).toBe(true)
    fireEvent.click(button)
    expect(open).not.toHaveBeenCalled()
  })
})
