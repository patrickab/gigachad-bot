import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { readFileSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import type { ReactNode } from "react"

const deck = vi.hoisted(() => ({
  /** Annotations of each page, by page number. */
  annotations: new Map<number, unknown[]>(),
}))

vi.mock("react-pdf", async () => {
  // vi.mock runs before static imports, so the factory loads React itself
  const React = await import("react")
  const pdf = {
    numPages: 3,
    getPage: async (pageNumber: number) => ({
      getViewport: () => ({ width: 160, height: 90 }),
      getAnnotations: async () => deck.annotations.get(pageNumber) ?? [],
    }),
  }
  function Document({ onLoadSuccess, children, className }: { onLoadSuccess?: (loaded: unknown) => void; children?: ReactNode; className?: string }) {
    const onLoadRef = React.useRef(onLoadSuccess)
    React.useEffect(() => { onLoadRef.current?.(pdf) }, [])
    return <div className={className}>{children}</div>
  }
  function Page({ pageNumber, onRenderSuccess }: { pageNumber: number; onRenderSuccess?: () => void }) {
    const onRenderRef = React.useRef(onRenderSuccess)
    React.useLayoutEffect(() => { onRenderRef.current = onRenderSuccess })
    React.useEffect(() => { onRenderRef.current?.() }, [pageNumber])
    return <canvas data-testid={`page ${pageNumber}`} className="react-pdf__Page__canvas" />
  }
  return { Document, Page, pdfjs: { GlobalWorkerOptions: {} } }
})

import { PresentationWindow } from "../PresentationWindow"
import { parsePresentQuery } from "../presentParams"

const offsetWidth = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "offsetWidth")
const offsetHeight = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "offsetHeight")

beforeAll(() => {
  Object.defineProperty(HTMLElement.prototype, "offsetWidth", { configurable: true, get: () => 800 })
  Object.defineProperty(HTMLElement.prototype, "offsetHeight", { configurable: true, get: () => 600 })
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} })
})

afterAll(() => {
  if (offsetWidth) Object.defineProperty(HTMLElement.prototype, "offsetWidth", offsetWidth)
  if (offsetHeight) Object.defineProperty(HTMLElement.prototype, "offsetHeight", offsetHeight)
  vi.unstubAllGlobals()
})

afterEach(() => {
  cleanup()
  deck.annotations.clear()
})

let channelCount = 0
/** A channel no other test's windows listen on. */
const freshChannel = () => `test-${++channelCount}`

/** The page numbers drawn on screen in `region`. */
function shownPages(region: HTMLElement): number[] {
  return within(region).queryAllByTestId(/^page /).map((canvas) => Number(canvas.dataset.testid?.slice("page ".length)))
}

const press = (key: string) => fireEvent.keyDown(window, { key })

describe("PresentationWindow", () => {
  it("moves through the slides by key and stays within the deck", async () => {
    render(<PresentationWindow pdfUrl="/deck.pdf" role="audience" channel={freshChannel()} />)
    const slide = await screen.findByRole("main", { name: "Slide" })
    await waitFor(() => expect(shownPages(slide)).toEqual([1]))

    const expectations: [string, number][] = [
      ["ArrowRight", 2], ["ArrowDown", 3], ["ArrowRight", 3], [" ", 3], ["PageDown", 3],
      ["ArrowLeft", 2], ["ArrowUp", 1], ["PageUp", 1], ["End", 3], ["Enter", 3], ["Home", 1], [" ", 2],
    ]
    for (const [key, page] of expectations) {
      press(key)
      await waitFor(() => expect(shownPages(slide), `after ${key}`).toEqual([page]))
    }
  })

  it("shows the presenter the slide count, the next slide, End after the last one, and the notes of the current slide's Text annotations only", async () => {
    deck.annotations.set(1, [
      { subtype: "Text", contentsObj: { str: "Greet the room" } },
      { subtype: "Link", contents: "not a note" },
      { subtype: "Popup", contentsObj: { str: "not a note either" } },
      { subtype: "Text", contents: " Show the agenda " },
      { subtype: "Text", contents: "  " },
    ])
    deck.annotations.set(2, [{ subtype: "Text", contents: "Second slide note" }])
    render(<PresentationWindow pdfUrl="/deck.pdf" role="presenter" channel={freshChannel()} />)

    const notes = await screen.findByRole("region", { name: "Notes" })
    await waitFor(() => expect(notes.textContent).toBe("Greet the room\nShow the agenda"))
    expect(screen.getByText("1 / 3")).toBeTruthy()
    expect(shownPages(screen.getByRole("region", { name: "Current slide" }))).toEqual([1])
    expect(shownPages(screen.getByRole("region", { name: "Next slide" }))).toEqual([2])
    expect(screen.getByRole("region", { name: "Next slide" }).textContent).not.toContain("End")

    press("ArrowRight")
    await waitFor(() => expect(notes.textContent).toBe("Second slide note"))
    expect(screen.getByText("2 / 3")).toBeTruthy()

    press("End")
    await waitFor(() => expect(screen.getByText("3 / 3")).toBeTruthy())
    await waitFor(() => expect(notes.textContent).toBe(""))
    expect(shownPages(screen.getByRole("region", { name: "Next slide" }))).toEqual([])
    expect(screen.getByRole("region", { name: "Next slide" }).textContent).toBe("End")
  })

  it("opens the audience window on the presenter's channel by button and by A, then hides the button", async () => {
    const audience = { closed: false, focus: vi.fn() }
    const open = vi.spyOn(window, "open").mockReturnValue(audience as unknown as Window)
    const channel = freshChannel()
    render(<PresentationWindow pdfUrl="/deck.pdf" role="presenter" channel={channel} />)

    fireEvent.click(await screen.findByRole("button", { name: "Open audience window" }))

    expect(open).toHaveBeenCalledTimes(1)
    const url = new URL(String(open.mock.calls[0][0]), window.location.origin)
    expect(parsePresentQuery(url.search)).toEqual({ pdfUrl: "/deck.pdf", role: "audience", channel })
    expect(screen.queryByRole("button", { name: "Open audience window" })).toBeNull()

    press("a")
    expect(open).toHaveBeenCalledTimes(1)
    expect(audience.focus).toHaveBeenCalled()
    open.mockRestore()
  })

  it("opens the audience window with A, and the audience window never opens another", async () => {
    const open = vi.spyOn(window, "open").mockReturnValue({ closed: false, focus: vi.fn() } as unknown as Window)
    const channel = freshChannel()
    render(<PresentationWindow pdfUrl="/deck.pdf" role="audience" channel={channel} />)
    await screen.findByRole("main", { name: "Slide" })
    press("A")
    expect(open).not.toHaveBeenCalled()
    cleanup()

    render(<PresentationWindow pdfUrl="/deck.pdf" role="presenter" channel={channel} />)
    await screen.findByRole("button", { name: "Open audience window" })
    press("A")
    expect(open).toHaveBeenCalledTimes(1)
    expect(parsePresentQuery(new URL(String(open.mock.calls[0][0]), window.location.origin).search)?.role).toBe("audience")
    open.mockRestore()
  })

  it("draws every slide in true colours in the dark theme, while a workbench page canvas stays inverted", async () => {
    // The dark-theme rules of globals.css that filter a PDF canvas, in file order
    const css = readFileSync(resolve(dirname(fileURLToPath(import.meta.url)), "../../../../app/globals.css"), "utf8")
    const canvasRules = css.split("\n").filter((line) => line.includes("react-pdf__Page__canvas") && line.includes("filter:"))
    expect(canvasRules.length).toBeGreaterThan(0)
    const style = document.head.appendChild(document.createElement("style"))
    style.textContent = canvasRules.join("\n")
    const workbench = document.body.appendChild(document.createElement("canvas"))
    workbench.className = "react-pdf__Page__canvas"
    try {
      expect(getComputedStyle(workbench).filter).toBe("invert(1) hue-rotate(180deg)")

      const expectTrueColours = (canvases: HTMLElement[]) => {
        expect(canvases.length).toBeGreaterThan(0)
        for (const canvas of canvases) {
          expect(canvas.closest(".pdf-true-colors")).not.toBeNull()
          expect(getComputedStyle(canvas).filter).toBe("none")
        }
      }

      render(<PresentationWindow pdfUrl="/deck.pdf" role="presenter" channel={freshChannel()} />)
      await waitFor(() => expect(shownPages(screen.getByRole("region", { name: "Next slide" }))).toEqual([2]))
      expectTrueColours(screen.getAllByTestId(/^page /))
      cleanup()

      render(<PresentationWindow pdfUrl="/deck.pdf" role="audience" channel={freshChannel()} />)
      const slide = await screen.findByRole("main", { name: "Slide" })
      await waitFor(() => expect(shownPages(slide)).toEqual([1]))
      expectTrueColours(within(slide).getAllByTestId(/^page /))
    } finally {
      style.remove()
      workbench.remove()
    }
  })
})
