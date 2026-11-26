import { fireEvent, render, screen } from "@testing-library/react"
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

function Column({ turns, reveal = null }: { turns: TestTurn[]; reveal?: ConversationReveal | null }) {
  return (
    <ConversationColumn
      turns={turns}
      renderPreview={(turn) => <p>{turn.answer}</p>}
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
