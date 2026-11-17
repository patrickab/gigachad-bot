import { afterEach, describe, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import type { ComponentProps, ReactNode } from "react"
import type { PdfWord } from "@/components/content/pdf/pdfText"

// The viewer is out of scope: it lays out two pages, draws the pane's overlay over each and reports a double-click on them
vi.mock("@/components/content/LazyPdfViewer", () => ({
  LazyPdfViewer: ({ renderPageOverlay, onPageDoubleClick }: {
    renderPageOverlay?: (page: number, pageWidth: number, pageCount: number) => ReactNode
    onPageDoubleClick?: (page: number, x: number, y: number, word: PdfWord) => void
  }) => (
    <div onDoubleClick={() => onPageDoubleClick?.(1, 0, 0, { text: "", offset: 0, occurrence: 0 })}>
      {[1, 2].map((page) => <div key={page}>{renderPageOverlay?.(page, 600, 2)}</div>)}
    </div>
  ),
}))

import { LatexPdfPane } from "../LatexPdfPane"

afterEach(cleanup)

/** The pane showing a built PDF with `slideGap`. */
function paneOf(slideGap?: ComponentProps<typeof LatexPdfPane>["slideGap"], onPageDoubleClick = vi.fn()) {
  return (
    <LatexPdfPane
      url="slides.pdf"
      documentKey="slides"
      building={false}
      emptyMessage={null}
      pageBox={{ width: 600, top: 0, bottom: 0 }}
      onPageDoubleClick={onPageDoubleClick}
      slideGap={slideGap}
    />
  )
}

describe("LatexPdfPane slide gaps", () => {
  it("offers the gap between two pages, marks the marked one and keeps a double-click there off the page", () => {
    const onChoose = vi.fn<(page: number) => void>()
    const onPageDoubleClick = vi.fn()
    const view = render(paneOf({ marked: null, onChoose, onClear: vi.fn() }, onPageDoubleClick))

    // Two pages, one gap between them and none below the last
    const gap = screen.getByRole("button", { name: "Gap after page 1" })
    expect(screen.queryByRole("button", { name: "Gap after page 2" })).toBeNull()
    expect(gap).toHaveAttribute("aria-pressed", "false")
    fireEvent.click(gap)
    fireEvent.doubleClick(gap)
    expect(onChoose).toHaveBeenCalledWith(1)
    expect(onPageDoubleClick).not.toHaveBeenCalled()

    view.rerender(paneOf({ marked: 1, onChoose, onClear: vi.fn() }, onPageDoubleClick))
    expect(screen.getByRole("button", { name: "Gap after page 1" })).toHaveAttribute("aria-pressed", "true")
  })

  it("clears a marked gap on Escape", () => {
    const onClear = vi.fn()
    render(paneOf({ marked: 1, onChoose: vi.fn(), onClear }))

    fireEvent.keyDown(screen.getByRole("button", { name: "Gap after page 1" }), { key: "Escape" })

    expect(onClear).toHaveBeenCalledOnce()
  })

  it("offers no gap without slideGap", () => {
    render(paneOf())

    expect(screen.queryByRole("button", { name: /^Gap after/ })).toBeNull()
  })
})
