import { fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { PillPopover, usePillPopover } from "@/components/ui/PillPopover"

function Popover({ onClose }: { onClose: () => void }) {
  const popover = usePillPopover(onClose)
  return <PillPopover popover={popover} triggerLabel="Model" triggerContent="Model" label="Models">content</PillPopover>
}

/** Document mousedown and keydown listeners attached now, counted from the spies' calls. */
type Spy = { mock: { calls: unknown[][] } }
function documentListeners(add: Spy, remove: Spy) {
  const count = (spy: Spy) => spy.mock.calls.filter(([type]) => type === "mousedown" || type === "keydown").length
  return count(add) - count(remove)
}

afterEach(() => vi.restoreAllMocks())

describe("usePillPopover", () => {
  it("listens to the document only while open", () => {
    const add = vi.spyOn(document, "addEventListener")
    const remove = vi.spyOn(document, "removeEventListener")
    const onClose = vi.fn()
    render(<Popover onClose={onClose} />)
    expect(documentListeners(add, remove)).toBe(0)
    fireEvent.mouseDown(document.body)
    fireEvent.keyDown(document, { key: "Escape" })
    expect(onClose).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole("button", { name: "Model" }))
    expect(documentListeners(add, remove)).toBe(2)
  })

  it("closes on an outside press and on Escape while open", () => {
    const onClose = vi.fn()
    render(<Popover onClose={onClose} />)
    const trigger = screen.getByRole("button", { name: "Model" })
    fireEvent.click(trigger)
    fireEvent.mouseDown(document.body)
    expect(onClose).toHaveBeenCalledTimes(1)
    expect(trigger).toHaveAttribute("aria-expanded", "false")

    fireEvent.click(trigger)
    fireEvent.keyDown(document, { key: "Escape" })
    expect(onClose).toHaveBeenCalledTimes(2)
    expect(trigger).toHaveAttribute("aria-expanded", "false")
  })
})
