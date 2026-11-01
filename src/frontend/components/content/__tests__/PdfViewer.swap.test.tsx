import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest"
import { act, cleanup, render, screen, waitFor } from "@testing-library/react"
import type { ReactNode } from "react"

const pdf = vi.hoisted(() => ({
  /** Loads a mounted Document by its file. */
  load: new Map<string, () => void>(),
  /** Finishes drawing a mounted page, keyed `file#page`. */
  draw: new Map<string, () => void>(),
}))

vi.mock("react-pdf", async () => {
  // vi.mock runs before static imports, so the factory loads React itself
  const React = await import("react")
  const FileContext = React.createContext("")
  const openDocument = () => ({ numPages: 2, getPage: async () => ({ getViewport: () => ({ width: 100, height: 141 }) }) })
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
  function Page({ pageNumber, onRenderSuccess }: { pageNumber: number; onRenderSuccess?: () => void }) {
    const file = React.useContext(FileContext)
    const onRenderRef = React.useRef(onRenderSuccess)
    React.useLayoutEffect(() => { onRenderRef.current = onRenderSuccess })
    React.useEffect(() => {
      const key = `${file}#${pageNumber}`
      pdf.draw.set(key, () => onRenderRef.current?.())
      return () => { pdf.draw.delete(key) }
    }, [file, pageNumber])
    return <canvas data-testid={`${file}#${pageNumber}`} />
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
})

async function load(file: string) {
  await waitFor(() => expect(pdf.load.has(file)).toBe(true))
  await act(async () => pdf.load.get(file)?.())
  await waitFor(() => expect(pdf.draw.has(`${file}#2`)).toBe(true))
}

async function draw(file: string, page: number) {
  await act(async () => pdf.draw.get(`${file}#${page}`)?.())
}

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
