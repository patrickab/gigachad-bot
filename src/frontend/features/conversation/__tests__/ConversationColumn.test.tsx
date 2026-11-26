import { act, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { ConversationColumn, TURN_PAGE_SIZE, type ConversationReveal, type ConversationTurn } from "../ConversationColumn"

interface TestTurn extends ConversationTurn { answer: string }

function turnsOf(count: number, { live = false, streaming = "" } = {}): TestTurn[] {
  return Array.from({ length: count }, (_, index) => ({
    key: `k${index}`,
    label: `Prompt ${index}`,
    live: live && index === count - 1,
    answer: `Answer ${index}${index === count - 1 ? streaming : ""}`,
  }))
}

function Column({ turns, reveal = null, followSeq, after, readOnly }: { turns: TestTurn[]; reveal?: ConversationReveal | null; followSeq?: number; after?: string; readOnly?: boolean }) {
  return (
    <ConversationColumn
      turns={turns}
      followSeq={followSeq}
      readOnly={readOnly}
      renderAfter={(turn) => (turn.key === after ? <hr aria-label="After" /> : null)}
      renderPreview={(turn) => (turn.answer ? <p>{turn.answer}</p> : null)}
      previewText={(turn) => turn.answer}
      renderBody={(turn, _index, collapse) => (
        <>
          <p>{turn.label}</p>
          {collapse && <button type="button" onClick={collapse}>Collapse</button>}
          <p>{turn.answer}</p>
        </>
      )}
      reveal={reveal}
    />
  )
}

// jsdom has no layout: content height follows the text, the viewport is 100px, scrollTop is stored as set
const scrollTops = new WeakMap<Element, number>()
beforeEach(() => {
  vi.spyOn(HTMLElement.prototype, "scrollHeight", "get").mockImplementation(function (this: HTMLElement) { return (this.textContent ?? "").length * 10 })
  vi.spyOn(HTMLElement.prototype, "clientHeight", "get").mockImplementation(() => 100)
  Object.defineProperty(HTMLElement.prototype, "scrollTop", {
    configurable: true,
    get(this: HTMLElement) { return scrollTops.get(this) ?? 0 },
    set(this: HTMLElement, value: number) { scrollTops.set(this, value) },
  })
})
afterEach(() => {
  vi.restoreAllMocks()
  Reflect.deleteProperty(HTMLElement.prototype, "scrollTop")
})

describe("ConversationColumn window", () => {
  it("mounts the newest page of turns only and follows the end while streaming", () => {
    const { container, rerender } = render(<Column turns={turnsOf(45, { live: true })} />)
    const scroller = container.firstElementChild as HTMLElement
    const firstMounted = 45 - TURN_PAGE_SIZE
    expect(screen.queryByText(`Prompt ${firstMounted - 1}`)).toBeNull()
    expect(screen.getByText(`Prompt ${firstMounted}`)).toBeTruthy()
    expect(scroller.scrollTop).toBe(scroller.scrollHeight)

    rerender(<Column turns={turnsOf(45, { live: true, streaming: " Streaming a long answer" })} />)
    const grown = scroller.scrollHeight
    expect(scroller.textContent).toContain("Streaming a long answer")
    expect(scroller.scrollTop).toBe(grown)
  })

  it("keeps the read window while the reader is away from the end, and loads earlier turns on request", () => {
    const { container, rerender } = render(<Column turns={turnsOf(45)} />)
    const scroller = container.firstElementChild as HTMLElement
    scroller.scrollTop = 0
    fireEvent.scroll(scroller)

    rerender(<Column turns={turnsOf(60)} />)
    expect(screen.getByText(`Prompt ${45 - TURN_PAGE_SIZE}`)).toBeTruthy()
    expect(scroller.scrollTop).toBe(0)

    fireEvent.click(screen.getByRole("button", { name: "Show earlier turns" }))
    expect(screen.getByText(`Prompt ${45 - 2 * TURN_PAGE_SIZE}`)).toBeTruthy()
  })
})

describe("ConversationColumn open state", () => {
  it("opens the newest turn, keeps a live one open and keeps what the reader opened by key", () => {
    const { rerender } = render(<Column turns={turnsOf(3, { live: true })} />)
    expect(screen.queryByRole("button", { name: "Collapse" })).toBeNull()

    rerender(<Column turns={turnsOf(3)} />)
    fireEvent.click(screen.getByRole("button", { name: "Collapse" }))
    expect(screen.queryByRole("button", { name: "Collapse" })).toBeNull()
    fireEvent.click(screen.getByText("Prompt 1"))
    expect(screen.getByRole("button", { name: "Collapse" })).toBeTruthy()

    // The turn before it goes: the opened turn stays open, under its key
    rerender(<Column turns={turnsOf(3).filter((turn) => turn.key !== "k0")} />)
    const opened = document.querySelector("[data-turn-key='k1']")
    expect(opened?.querySelector(".gc-turn")?.getAttribute("data-expanded")).toBe("true")
  })

  it("reveals a turn outside the window: mounts, opens and scrolls to it once per seq", () => {
    const scrolled: Element[] = []
    Element.prototype.scrollIntoView = function (this: Element) { scrolled.push(this) }
    const turns = turnsOf(45)
    const { rerender } = render(<Column turns={turns} />)
    expect(screen.queryByText("Prompt 0")).toBeNull()

    rerender(<Column turns={turns} reveal={{ key: "k0", seq: 1 }} />)
    expect(document.querySelector("[data-turn-key='k0'] .gc-turn")?.getAttribute("data-expanded")).toBe("true")
    expect(scrolled.at(-1)?.getAttribute("data-turn-key")).toBe("k0")

    rerender(<Column turns={turns} reveal={{ key: "k0", seq: 1 }} />)
    expect(scrolled).toHaveLength(1)
    Reflect.deleteProperty(Element.prototype, "scrollIntoView")
  })
})

const isOpen = (key: string) => document.querySelector(`[data-turn-key='${key}'] .gc-turn`)?.getAttribute("data-expanded") === "true"
const isFocused = (key: string) => document.querySelector(`[data-turn-key='${key}'] .gc-turn`)?.hasAttribute("data-focused") ?? false

describe("ConversationColumn scrolling", () => {
  it("glides to the end for a new turn only, a streamed frame jumps", () => {
    const glides: ScrollToOptions[] = []
    HTMLElement.prototype.scrollTo = function (this: HTMLElement, options?: ScrollToOptions | number) { glides.push(options as ScrollToOptions) }
    const { container, rerender } = render(<Column turns={turnsOf(3)} />)
    const scroller = container.firstElementChild as HTMLElement
    expect(glides).toHaveLength(0)

    rerender(<Column turns={turnsOf(4, { live: true })} />)
    expect(glides).toEqual([{ top: scroller.scrollHeight, behavior: "smooth" }])

    rerender(<Column turns={turnsOf(4, { live: true, streaming: " more" })} />)
    expect(glides).toHaveLength(1)
    expect(scroller.scrollTop).toBe(scroller.scrollHeight)
    Reflect.deleteProperty(HTMLElement.prototype, "scrollTo")
  })

  it("keeps the end through late layout until the reader first interacts", () => {
    let resized = () => {}
    vi.stubGlobal("ResizeObserver", class { constructor(callback: () => void) { resized = callback } observe() {} disconnect() {} })
    const { container } = render(<Column turns={turnsOf(3)} />)
    const scroller = container.firstElementChild as HTMLElement
    scroller.scrollTop = 0
    resized()
    expect(scroller.scrollTop).toBe(scroller.scrollHeight)

    fireEvent.wheel(scroller)
    scroller.scrollTop = 0
    resized()
    expect(scroller.scrollTop).toBe(0)
    vi.unstubAllGlobals()
  })

  it("follows the end again on a new follow request", () => {
    const { container, rerender } = render(<Column turns={turnsOf(45)} followSeq={0} />)
    const scroller = container.firstElementChild as HTMLElement
    scroller.scrollTop = 0
    fireEvent.scroll(scroller)
    rerender(<Column turns={turnsOf(46)} followSeq={0} />)
    expect(scroller.scrollTop).toBe(0)

    rerender(<Column turns={turnsOf(46)} followSeq={1} />)
    expect(scroller.scrollTop).toBe(scroller.scrollHeight)
    expect(screen.queryByText(`Prompt ${45 - TURN_PAGE_SIZE}`)).toBeNull()
  })

  it("reveals a turn once it arrives", () => {
    Element.prototype.scrollIntoView = () => {}
    const { rerender } = render(<Column turns={[]} reveal={{ key: "k1", seq: 3 }} />)
    rerender(<Column turns={turnsOf(3)} reveal={{ key: "k1", seq: 3 }} />)
    expect(isOpen("k1")).toBe(true)
    Reflect.deleteProperty(Element.prototype, "scrollIntoView")
  })
})

describe("ConversationColumn rows and previews", () => {
  it("renders a row after its turn", () => {
    render(<Column turns={turnsOf(3)} after="k1" />)
    const row = screen.getByRole("separator", { name: "After" })
    expect(row.previousElementSibling?.getAttribute("data-turn-key")).toBe("k1")
  })

  it("renders a collapsed preview near the viewport only, or once the turn takes focus", () => {
    const observed = new Map<Element, IntersectionObserverCallback>()
    vi.stubGlobal("IntersectionObserver", class {
      constructor(private callback: IntersectionObserverCallback) {}
      observe(element: Element) { observed.set(element, this.callback) }
      disconnect() {}
    })
    render(<Column turns={turnsOf(3)} />)
    expect(screen.queryByText("Answer 0", { selector: "p" })).toBeNull()
    expect(screen.queryByText("Answer 1", { selector: "p" })).toBeNull()

    const pending = document.querySelector("[data-turn-key='k0'] [data-preview-pending]")!
    act(() => observed.get(pending)!([{ isIntersecting: true } as IntersectionObserverEntry], {} as IntersectionObserver))
    expect(screen.getByText("Answer 0", { selector: "p" })).toBeTruthy()

    fireEvent.focus(screen.getByRole("button", { name: "Prompt 1" }))
    expect(screen.getByText("Answer 1", { selector: "p" })).toBeTruthy()
    vi.unstubAllGlobals()
  })

  it("holds a waiting preview's text for screen readers, and reserves nothing for a turn without a preview", () => {
    vi.stubGlobal("IntersectionObserver", class { observe() {} disconnect() {} })
    render(<Column turns={[{ ...turnsOf(1)[0], key: "empty", answer: "" }, ...turnsOf(3)]} />)
    const pending = document.querySelector("[data-turn-key='k0'] [data-preview-pending]")!
    expect(pending.querySelector(".sr-only")?.textContent).toBe("Answer 0")
    expect(document.querySelector("[data-turn-key='empty'] [data-preview-pending]")).toBeNull()
    vi.unstubAllGlobals()
  })

  it("renders an already visible preview at once", () => {
    vi.stubGlobal("IntersectionObserver", class { observe() {} disconnect() {} })
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue({ top: 100, bottom: 196, left: 0, right: 600, width: 600, height: 96, x: 0, y: 100, toJSON() {} })
    render(<Column turns={turnsOf(2)} />)
    expect(screen.getByText("Answer 0")).toBeTruthy()
    vi.unstubAllGlobals()
  })
})

describe("ConversationColumn keyboard focus", () => {
  const key = (init: KeyboardEventInit, target: EventTarget = document.body) => act(() => { target.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, cancelable: true, ...init })) })

  it("moves with Ctrl+Up/Down, Enter opens or collapses the focused turn, Escape drops the focus", () => {
    render(<Column turns={turnsOf(4)} />)
    key({ key: "ArrowUp", ctrlKey: true })
    expect(isFocused("k3")).toBe(true)
    key({ key: "ArrowUp", ctrlKey: true })
    expect(isFocused("k2")).toBe(true)
    expect(isFocused("k3")).toBe(false)

    key({ key: "Enter" })
    expect(isOpen("k2")).toBe(true)
    expect(isFocused("k2")).toBe(false)

    key({ key: "ArrowDown", ctrlKey: true })
    expect(isFocused("k0")).toBe(true)
    key({ key: "Escape" })
    expect(isFocused("k0")).toBe(false)
  })

  it("mounts an earlier turn the focus moves to, and ignores keys from outside its surface", () => {
    const outside = document.createElement("input")
    document.body.append(outside)
    const { container } = render(<Column turns={turnsOf(TURN_PAGE_SIZE + 5)} />)
    key({ key: "ArrowUp", ctrlKey: true }, outside)
    expect(container.querySelector(".gc-turn[data-focused]")).toBeNull()

    for (let step = 0; step < TURN_PAGE_SIZE + 1; step++) key({ key: "ArrowUp", ctrlKey: true })
    expect(isFocused("k4")).toBe(true)
    expect(screen.getByText("Prompt 1")).toBeTruthy()
    outside.remove()
  })

  it("moves from a control elsewhere in the tab, e.g. the sidebar, where Enter stays the control's", () => {
    const sidebar = document.createElement("button")
    document.body.append(sidebar)
    render(<Column turns={turnsOf(3)} />)
    key({ key: "ArrowUp", ctrlKey: true }, sidebar)
    expect(isFocused("k2")).toBe(true)
    key({ key: "Enter" }, sidebar)
    expect(isFocused("k2")).toBe(true)
    sidebar.remove()
  })

  it("never reaches a read-only column", () => {
    render(<Column turns={turnsOf(3)} readOnly />)
    key({ key: "ArrowUp", ctrlKey: true })
    expect(document.querySelector(".gc-turn[data-focused]")).toBeNull()
  })

  it("reaches one column only: the one whose surface holds the key, else the first", () => {
    const sidebar = document.createElement("button")
    document.body.append(sidebar)
    render(<><div data-testid="first"><Column turns={turnsOf(3)} /></div><div data-testid="second"><Column turns={turnsOf(3)} /></div></>)
    const focusedIn = (id: string) => screen.getByTestId(id).querySelectorAll(".gc-turn[data-focused]").length
    const composer = document.createElement("textarea")
    screen.getByTestId("second").append(composer)
    key({ key: "ArrowUp", ctrlKey: true }, composer)
    expect([focusedIn("first"), focusedIn("second")]).toEqual([0, 1])

    key({ key: "ArrowUp", ctrlKey: true }, sidebar)
    expect(focusedIn("first") + focusedIn("second")).toBe(2)
    sidebar.remove()
  })

  it("leaves a key inside a read-only column to it, which takes none", () => {
    render(<><div><Column turns={turnsOf(3)} /></div><div data-testid="preview"><Column turns={turnsOf(3)} readOnly /></div></>)
    const inPreview = screen.getByTestId("preview").querySelector("[data-turn-key='k2']")!
    key({ key: "ArrowUp", ctrlKey: true }, inPreview)
    expect(document.querySelector(".gc-turn[data-focused]")).toBeNull()
  })
})
