import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { Settings } from "lucide-react"
import { PopoverSelector } from "@/components/ui/PopoverSelector"

const CATEGORIES = [
  { key: "fruit", label: "Fruit", items: ["Apple", "Pear", "Plum"] },
  { key: "vegetables", label: "Vegetables", items: ["Leek"] },
]

/** The selector over `CATEGORIES` as tabs, opened by a click, with `selected` picked (none by default). */
function renderSelector({ selected, onSettings }: { selected?: string; onSettings?: () => void } = {}) {
  const onSelect = vi.fn()
  render(<PopoverSelector<string>
    triggerLabel="Pick"
    triggerContent="Pick"
    label="Produce"
    categories={CATEGORIES}
    itemKey={(item) => item}
    isSelected={(item) => item === selected}
    renderItem={(item) => <span className="flex-1 truncate">{item}</span>}
    onSelect={onSelect}
    headerAction={onSettings && { icon: Settings, label: "Settings", onClick: onSettings }}
  />)
  const trigger = screen.getByRole("button", { name: "Pick" })
  fireEvent.click(trigger)
  return { trigger, onSelect }
}

/** Enter on `element` as a browser does it: a click on the button unless a handler took the key. */
function pressEnter(element: Element) {
  if (fireEvent.keyDown(element, { key: "Enter" })) fireEvent.click(element)
}

const row = (name: string) => screen.getByRole("menuitemradio", { name })

describe("PopoverSelector", () => {
  it("shows the categories as tabs, one list at a time", () => {
    renderSelector()

    expect(screen.getAllByRole("menuitemradio").map((item) => item.textContent)).toEqual(["Apple", "Pear", "Plum"])
    fireEvent.click(screen.getByRole("button", { name: "Vegetables" }))

    expect(screen.getAllByRole("menuitemradio").map((item) => item.textContent)).toEqual(["Leek"])
  })

  it("moves focus with the arrows, wrapping around, and picks the focused row with Enter", async () => {
    const { trigger, onSelect } = renderSelector()

    fireEvent.keyDown(trigger, { key: "ArrowDown" })
    for (const name of ["Apple", "Pear", "Plum"]) fireEvent.keyDown(row(name), { key: "ArrowDown" })
    expect(row("Apple")).toHaveFocus()
    fireEvent.keyDown(row("Apple"), { key: "ArrowUp" })
    expect(row("Plum")).toHaveFocus()
    pressEnter(row("Plum"))

    expect(onSelect).toHaveBeenCalledExactlyOnceWith("Plum")
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    expect(trigger).toHaveFocus()
  })

  it("moves focus from the trigger to the selected row first", () => {
    const { trigger } = renderSelector({ selected: "Pear" })

    fireEvent.keyDown(trigger, { key: "ArrowDown" })

    expect(row("Pear")).toHaveFocus()
  })

  it("picks the focused row with Enter, not the selected one", () => {
    const { onSelect } = renderSelector({ selected: "Apple" })

    row("Plum").focus()
    pressEnter(row("Plum"))

    expect(onSelect).toHaveBeenCalledExactlyOnceWith("Plum")
  })

  it("runs the focused header action with Enter without picking a row", () => {
    const onSettings = vi.fn()
    const { onSelect } = renderSelector({ selected: "Apple", onSettings })

    const settings = screen.getByRole("button", { name: "Settings" })
    settings.focus()
    pressEnter(settings)

    expect(onSettings).toHaveBeenCalledOnce()
    expect(onSelect).not.toHaveBeenCalled()
  })

  it("closes with Escape without picking anything, focus back on the trigger", async () => {
    const { trigger, onSelect } = renderSelector()

    fireEvent.keyDown(trigger, { key: "ArrowDown" })
    fireEvent.keyDown(row("Apple"), { key: "Escape" })

    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    expect(onSelect).not.toHaveBeenCalled()
    expect(trigger).toHaveFocus()
  })
})
