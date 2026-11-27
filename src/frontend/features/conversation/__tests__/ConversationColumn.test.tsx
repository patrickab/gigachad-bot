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

describe("ConversationColumn keyboard focus", () => {
  const key = (init: KeyboardEventInit, target: EventTarget = document.body) => act(() => { target.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, cancelable: true, ...init })) })

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
})
