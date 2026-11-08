import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import type { ReactNode } from "react"

const pdf = vi.hoisted(() => ({
  /** Loads a mounted Document by its file. */
  load: new Map<string, () => void>(),
  /** Finishes drawing a mounted page, keyed `file#page`. */
  draw: new Map<string, () => void>(),
  /** The page's text items in place of the default ones. */
  text: null as unknown[] | null,
  /** The annotations every page reads. */
  annotations: [] as { id: string; subtype: string; annotationFlags: number }[],
}))

vi.mock("react-pdf", async () => {
  // vi.mock runs before static imports, so the factory loads React itself
  const React = await import("react")
  const FileContext = React.createContext("")
  // Page text 10 pt tall: "Growth doubled" on a baseline 100 pt down a 141 pt page, "Second body" on one 40 pt down
  const textItem = (str: string, baseline: number) => ({ str, transform: [10, 0, 0, 10, 20, baseline], width: 50, height: 10 })
  const page = {
    getViewport: () => ({ width: 100, height: 141, convertToViewportPoint: (x: number, y: number) => [x, 141 - y] }),
    getTextContent: async () => ({ items: pdf.text ?? [textItem("Growth doubled", 41), { type: "beginMarkedContent" }, textItem("Second body", 101)] }),
  }
  const openDocument = () => ({ numPages: 2, getPage: async () => page })
  function Document({ file, onLoadSuccess, children }: { file: string; onLoadSuccess?: (pdf: unknown) => void; children?: ReactNode }) {
    const [loaded, setLoaded] = React.useState(false)
    const onLoadRef = React.useRef(onLoadSuccess)
    React.useLayoutEffect(() => { onLoadRef.current = onLoadSuccess })
    React.useEffect(() => {
      pdf.load.set(file, () => {
        setLoaded(true)
        onLoadRef.current?.(openDocument())
      })
      return () => { pdf.load.delete(file) }
    }, [file])
    return <div>{loaded && <FileContext.Provider value={file}>{children}</FileContext.Provider>}</div>
  }
  function Page({ pageNumber, onRenderSuccess, onGetAnnotationsSuccess }: { pageNumber: number; onRenderSuccess?: () => void; onGetAnnotationsSuccess?: (annotations: unknown[]) => void }) {
    const file = React.useContext(FileContext)
    const onRenderRef = React.useRef(onRenderSuccess)
    React.useLayoutEffect(() => { onRenderRef.current = onRenderSuccess })
    // Like react-pdf, the annotation layer draws the very array `onGetAnnotationsSuccess` got
    const [annotations] = React.useState(() => {
      const list = pdf.annotations.map((annotation) => ({ ...annotation }))
      onGetAnnotationsSuccess?.(list)
      return list
    })
    React.useEffect(() => {
      const key = `${file}#${pageNumber}`
      pdf.draw.set(key, () => onRenderRef.current?.())
      return () => { pdf.draw.delete(key) }
    }, [file, pageNumber])
    // The canvas, like react-pdf's text layer the page's text, and its annotation layer
    return (
      <>
        <canvas data-testid={`${file}#${pageNumber}`} />
        <span data-testid={`${file}#${pageNumber} text`}>Second body paragraph</span>
        {annotations.map((annotation) => <i key={annotation.id} data-testid={`${file}#${pageNumber} annotation`} data-subtype={annotation.subtype} />)}
      </>
    )
  }
  return { Document, Page, pdfjs: { GlobalWorkerOptions: {} } }
})

import { PdfViewer } from "../PdfViewer"

const clientWidth = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "clientWidth")

beforeAll(() => {
  Object.defineProperty(HTMLElement.prototype, "clientWidth", { configurable: true, get: () => 800 })
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} })
  vi.stubGlobal("matchMedia", () => ({ matches: false, addEventListener: vi.fn(), removeEventListener: vi.fn() }))
  // Every page is near the viewport
  vi.stubGlobal("IntersectionObserver", class {
    constructor(private callback: IntersectionObserverCallback) {}
    observe(target: Element) {
      queueMicrotask(() => this.callback([{ target, isIntersecting: true, rootBounds: { height: 1 } } as unknown as IntersectionObserverEntry], this as unknown as IntersectionObserver))
    }
    unobserve() {}
    disconnect() {}
  })
})

afterAll(() => {
  if (clientWidth) Object.defineProperty(HTMLElement.prototype, "clientWidth", clientWidth)
  vi.unstubAllGlobals()
})

afterEach(() => {
  cleanup()
  pdf.load.clear()
  pdf.draw.clear()
  pdf.text = null
  pdf.annotations = []
})

async function load(file: string) {
  await waitFor(() => expect(pdf.load.has(file)).toBe(true))
  await act(async () => pdf.load.get(file)?.())
  await waitFor(() => expect(pdf.draw.has(`${file}#2`)).toBe(true))
}

async function draw(file: string, page: number) {
  await act(async () => pdf.draw.get(`${file}#${page}`)?.())
}

describe("PdfViewer annotations", () => {
  it("draws links but neither hidden annotations nor Text note icons, like pdfpc's notes", async () => {
    pdf.annotations = [
      { id: "note", subtype: "Text", annotationFlags: 6 },
      { id: "link", subtype: "Link", annotationFlags: 0 },
      { id: "hidden", subtype: "Square", annotationFlags: 2 },
      { id: "noview", subtype: "Link", annotationFlags: 32 },
      { id: "printed", subtype: "Square", annotationFlags: 4 },
    ]
    render(<PdfViewer url="a.pdf" />)
    await load("a.pdf")

    expect(screen.getAllByTestId("a.pdf#1 annotation").map((annotation) => annotation.dataset.subtype)).toEqual(["Link", "Square"])
  })
})

describe("PdfViewer url swap", () => {
  it("keeps the shown pages until the new url's pages near the viewport have drawn", async () => {
    const onPageRendered = vi.fn()
    const { rerender } = render(<PdfViewer url="a.pdf" documentKey="paper" onPageRendered={onPageRendered} />)
    await load("a.pdf")
    await draw("a.pdf", 1)
    expect(onPageRendered).toHaveBeenLastCalledWith("a.pdf")

    rerender(<PdfViewer url="b.pdf" documentKey="paper" onPageRendered={onPageRendered} />)
    await load("b.pdf")
    await draw("b.pdf", 1)
    expect(screen.getByTestId("a.pdf#1")).toBeInTheDocument()
    expect(screen.getByTestId("b.pdf#1").closest("[inert]")).not.toBeNull()
    expect(onPageRendered).not.toHaveBeenCalledWith("b.pdf")

    await draw("b.pdf", 2)
    await waitFor(() => expect(screen.queryByTestId("a.pdf#1")).toBeNull())
    expect(screen.getByTestId("b.pdf#1").closest("[inert]")).toBeNull()
    expect(onPageRendered).toHaveBeenLastCalledWith("b.pdf")
  })

  it("replaces a pending url with a newer one", async () => {
    const { rerender } = render(<PdfViewer url="a.pdf" documentKey="paper" />)
    await load("a.pdf")
    rerender(<PdfViewer url="b.pdf" documentKey="paper" />)
    await load("b.pdf")
    rerender(<PdfViewer url="c.pdf" documentKey="paper" />)
    expect(screen.queryByTestId("b.pdf#1")).toBeNull()
    expect(screen.getByTestId("a.pdf#1")).toBeInTheDocument()

    await load("c.pdf")
    await draw("c.pdf", 1)
    await draw("c.pdf", 2)
    await waitFor(() => expect(screen.queryByTestId("a.pdf#1")).toBeNull())
    expect(screen.getByTestId("c.pdf#1")).toBeInTheDocument()
  })

  it("drops the old pages at once without a documentKey", async () => {
    const { rerender } = render(<PdfViewer url="a.pdf" />)
    await load("a.pdf")
    rerender(<PdfViewer url="b.pdf" />)
    expect(screen.queryByTestId("a.pdf#1")).toBeNull()
  })
})

describe("PdfViewer double-click", () => {
  /** Page 2 of a.pdf, its 100 x 141 pt drawn at 400 px wide, 4 px per point. */
  async function showPage() {
    const onPageDoubleClick = vi.fn()
    render(<PdfViewer url="a.pdf" onPageDoubleClick={onPageDoubleClick} />)
    await load("a.pdf")
    const canvas = screen.getByTestId("a.pdf#2")
    vi.spyOn(canvas.closest<HTMLElement>(".pdf-page")!, "getBoundingClientRect").mockReturnValue(new DOMRect(50, 200, 400, 564))
    return { onPageDoubleClick, canvas }
  }

  /** Selects characters `start` to `end` of the text of `file#page`, as a double-click on a word does. */
  function select(file: string, page: number, start: number, end: number) {
    const range = document.createRange()
    const text = screen.getByTestId(`${file}#${page} text`).firstChild!
    range.setStart(text, start)
    range.setEnd(text, end)
    window.getSelection()!.removeAllRanges()
    window.getSelection()!.addRange(range)
  }

  afterEach(() => window.getSelection()?.removeAllRanges())

  /** The word a double-click reports. */
  const word = (text: string, offset = 0, occurrence = 0) => ({ text, offset, occurrence })
  /** A text item on `baseline` from `left` pt, lines 10 pt tall, 2 pt per character. */
  const item = (str: string, baseline: number, left = 20) => ({ str, transform: [10, 0, 0, 10, left, baseline], width: 2 * str.length, height: 10, hasEOL: true })

  it("reports the page and the spot in PDF points from its top-left corner, without a word when nothing is selected and no text is near", async () => {
    const { onPageDoubleClick, canvas } = await showPage()

    fireEvent.doubleClick(canvas, { clientX: 410, clientY: 340 })

    // "Second body" ends 20 pt left of the spot, two line heights
    await waitFor(() => expect(onPageDoubleClick).toHaveBeenCalledWith(2, 90, 35, word("")))
  })

  it("reports the word the double-click selected on the page, without the space around it", async () => {
    const { onPageDoubleClick } = await showPage()
    select("a.pdf", 2, 6, 12)

    fireEvent.doubleClick(screen.getByTestId("a.pdf#2 text"), { clientX: 150, clientY: 340 })

    await waitFor(() => expect(onPageDoubleClick).toHaveBeenCalledWith(2, 25, 35, word("body")))
  })

  it("reports the clicked character of the selected word, measured in the text layer, and which of the equal words of its rendered line it is", async () => {
    pdf.text = [item("body and body", 101)]
    const { onPageDoubleClick } = await showPage()
    select("a.pdf", 2, 6, 12)
    // Text layer glyphs 10 px wide: character 9 of "Second body paragraph", the "d" of "body", spans 210 to 220 px
    const glyphBox = Object.getOwnPropertyDescriptor(Range.prototype, "getBoundingClientRect")
    Object.defineProperty(Range.prototype, "getBoundingClientRect", {
      configurable: true,
      value(this: Range) { return new DOMRect(120 + 10 * this.startOffset, 0, 10, 10) },
    })
    try {
      // 41 pt across, over the second "body" of the page's line
      fireEvent.doubleClick(screen.getByTestId("a.pdf#2 text"), { clientX: 214, clientY: 340 })

      await waitFor(() => expect(onPageDoubleClick).toHaveBeenCalledWith(2, 41, 35, word("body", 2, 1)))
    } finally {
      if (glyphBox) Object.defineProperty(Range.prototype, "getBoundingClientRect", glyphBox)
      else delete (Range.prototype as Partial<Range>).getBoundingClientRect
    }
  })

  it("reports no word for a selection on another page", async () => {
    const { onPageDoubleClick, canvas } = await showPage()
    select("a.pdf", 1, 7, 11)

    fireEvent.doubleClick(canvas, { clientX: 410, clientY: 340 })

    await waitFor(() => expect(onPageDoubleClick).toHaveBeenCalledWith(2, 90, 35, word("")))
  })

  it.each([
    ["nothing", null],
    ["only the blank between two words", [6, 7]],
  ] as const)("reports the page's word nearest a double-click between lines that selected %s", async (_case, blank) => {
    // The evidence primitive's claim over its body: the body's top 5 pt below the claim's baseline
    pdf.text = [item("One short claim", 101), item("One or two sentences on", 86)]
    const { onPageDoubleClick, canvas } = await showPage()
    if (blank) select("a.pdf", 2, ...blank)

    // 44 pt down, 4 pt below the claim and 1 pt above the body, under its 16th character, the 5th of "sentences"
    fireEvent.doubleClick(canvas, { clientX: 250, clientY: 376 })

    await waitFor(() => expect(onPageDoubleClick).toHaveBeenCalledWith(2, 50, 44, word("sentences", 4)))
  })

  it.each([
    ["one text item", [item("see the first and the last", 101)]],
    ["two text items apart by a space", [item("see the first", 101), item("and the last", 101, 48), item("Title the", 121)]],
  ])("reports the character under the spot of a repeated nearest word and which of its rendered line it is, the line in %s", async (_case, text) => {
    pdf.text = text
    const { onPageDoubleClick, canvas } = await showPage()

    // 59 pt across, over the "h" of the second "the"
    fireEvent.doubleClick(canvas, { clientX: 286, clientY: 340 })

    await waitFor(() => expect(onPageDoubleClick).toHaveBeenCalledWith(2, 59, 35, word("the", 1, 1)))
  })
})

describe("PdfViewer reveal", () => {
  // 400 px wide pages of 100 x 141 pt: 564 px tall with a 12 px gap below, under 56 px of top space
  const pageBox = { width: 400, top: 56, bottom: 12 }

  it("scrolls the spot to a third of the scroller's height and marks it there, again for a new key", async () => {
    const view = render(<PdfViewer url="a.pdf" pageBox={pageBox} />)
    await load("a.pdf")
    const scroller = view.container.querySelector<HTMLElement>(".pdf-scroll")!
    Object.defineProperty(scroller, "clientHeight", { configurable: true, value: 600 })

    view.rerender(<PdfViewer url="a.pdf" pageBox={pageBox} reveal={{ page: 2, y: 70.5, key: 1 }} />)

    await waitFor(() => expect(scroller.scrollTop).toBe(56 + 564 + 12 + 282 - 200))
    const mark = view.container.querySelector<HTMLElement>(".pdf-reveal-mark")!
    expect(mark.closest(".pdf-page")).toHaveAttribute("data-page-number", "2")
    expect(mark.style.top).toBe("50%")

    scroller.scrollTop = 0
    view.rerender(<PdfViewer url="a.pdf" pageBox={pageBox} reveal={{ page: 2, y: 70.5, key: 2 }} />)
    await waitFor(() => expect(scroller.scrollTop).toBe(714))
  })

  it("waits for the document to lay out before revealing a spot asked for earlier", async () => {
    const view = render(<PdfViewer url="a.pdf" pageBox={pageBox} reveal={{ page: 2, y: 141, key: 1 }} />)
    const scroller = view.container.querySelector<HTMLElement>(".pdf-scroll")!
    Object.defineProperty(scroller, "clientHeight", { configurable: true, value: 600 })
    expect(scroller.scrollTop).toBe(0)

    await load("a.pdf")

    await waitFor(() => expect(scroller.scrollTop).toBe(56 + 564 + 12 + 564 - 200))
  })

  it.each([
    ["the first word in order the page's text holds, from the middle of its line", ["absent", "body", "Growth"], 70.5, 56 + 564 + 12 + 140 - 200],
    ["the spot when the page's text holds none of the words", ["absent"], 70.5, 56 + 564 + 12 + 282 - 200],
    ["the page's top when the spot is off the page as well", ["absent"], 500, 56 + 564 + 12 - 200],
  ])("scrolls to %s", async (_case, words, y, scrollTop) => {
    const view = render(<PdfViewer url="a.pdf" pageBox={pageBox} />)
    await load("a.pdf")
    const scroller = view.container.querySelector<HTMLElement>(".pdf-scroll")!
    Object.defineProperty(scroller, "clientHeight", { configurable: true, value: 600 })

    view.rerender(<PdfViewer url="a.pdf" pageBox={pageBox} reveal={{ page: 2, y, words, key: 1 }} />)

    await waitFor(() => expect(scroller.scrollTop).toBe(scrollTop))
  })

  it("marks a beamer body line on the line itself, not on the bottom of the line above", async () => {
    // Text as pdf.js reads a beamer frame: a title, an empty end-of-line item, then one item per body line 13.5 pt apart
    const item = (str: string, baseline: number, height = 10) => ({ str, transform: [10, 0, 0, 10, 20, baseline], width: 30, height, hasEOL: true })
    pdf.text = [item("Frame 1 Title", 131, 14), item("", 80, 0), item("alpha1", 80), item("beta1", 66.5), item("gamma1", 53)]
    const view = render(<PdfViewer url="a.pdf" pageBox={pageBox} />)
    await load("a.pdf")

    view.rerender(<PdfViewer url="a.pdf" pageBox={pageBox} reveal={{ page: 1, y: 0.99, words: ["beta"], key: 1 }} />)

    // beta1 spans 64.5 to 74.5 pt down the page, alpha1's baseline is 61 pt down
    await waitFor(() => expect(view.container.querySelector<HTMLElement>(".pdf-reveal-mark")?.style.top).toBe(`${(69.5 / 141) * 100}%`))
  })
})

describe("PdfViewer page overlay", () => {
  it("draws renderPageOverlay over every page, given the page's width and the page count", async () => {
    render(<PdfViewer url="a.pdf" pageBox={{ width: 600, top: 0, bottom: 0 }} renderPageOverlay={(page, pageWidth, pageCount) => <i data-testid="overlay">{`${page}/${pageCount} at ${pageWidth}`}</i>} />)
    await load("a.pdf")

    expect(screen.getAllByTestId("overlay").map((overlay) => overlay.textContent)).toEqual(["1/2 at 600", "2/2 at 600"])
  })
})
