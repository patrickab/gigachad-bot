import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { CommandPicker, pickerLayout } from "@/components/ui/CommandPicker"

const GROUPS = [
  { key: "fruit", label: "Fruit", items: ["Apple", "Pear"] },
  { key: "vegetables", label: "Vegetables", items: ["Leek"] },
]

/** The picker over `GROUPS`, opened by a click, with `onPick` succeeding by default and an optional preview window. */
function renderPicker({ onPick = vi.fn(() => true), renderPreview }: {
  onPick?: (item: string) => boolean
  renderPreview?: (item: string) => string | null
} = {}) {
  render(<CommandPicker<string>
    triggerLabel="Pick"
    triggerContent="Pick"
    label="Produce"
    groups={GROUPS}
    itemKey={(item) => item}
    renderItem={(item) => <span className="flex-1 truncate">{item}</span>}
    searchPlaceholder="Search produce"
    searchText={(item) => item}
    renderPreview={renderPreview}
    onPick={onPick}
  />)
  const trigger = screen.getByRole("button", { name: "Pick" })
  fireEvent.click(trigger)
  return { trigger, search: screen.getByRole("combobox", { name: "Search produce" }), onPick }
}

const option = (name: string) => screen.getByRole("option", { name })
const highlighted = () => screen.getByRole("option", { selected: true }).textContent
const card = () => document.body.querySelector("[aria-hidden].fixed")

describe("CommandPicker", () => {
  it("shows every group as a headed section of one list", () => {
    renderPicker()

    expect(screen.getAllByRole("group")).toEqual([screen.getByRole("group", { name: "Fruit" }), screen.getByRole("group", { name: "Vegetables" })])
    expect(within(screen.getByRole("group", { name: "Fruit" })).getAllByRole("option").map((item) => item.textContent)).toEqual(["Apple", "Pear"])
    expect(within(screen.getByRole("group", { name: "Vegetables" })).getAllByRole("option").map((item) => item.textContent)).toEqual(["Leek"])
  })

  it("opens with the search field focused and the first row highlighted", () => {
    const { search } = renderPicker()

    expect(search).toHaveFocus()
    expect(highlighted()).toBe("Apple")
    expect(search).toHaveAttribute("aria-activedescendant", option("Apple").id)
    expect(screen.getAllByRole("option", { selected: true })).toHaveLength(1)
  })

  it("moves the highlight across groups with the arrows, wrapping, jumps to the ends with Home and End, and picks it with Enter", async () => {
    const { trigger, search, onPick } = renderPicker()

    const moves: [string, string][] = [["ArrowDown", "Pear"], ["ArrowDown", "Leek"], ["ArrowDown", "Apple"], ["ArrowUp", "Leek"], ["Home", "Apple"], ["End", "Leek"]]
    for (const [key, name] of moves) {
      fireEvent.keyDown(search, { key })
      expect(highlighted()).toBe(name)
    }
    expect(search).toHaveFocus()
    fireEvent.keyDown(search, { key: "Enter" })

    expect(onPick).toHaveBeenCalledExactlyOnceWith("Leek")
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    expect(trigger).toHaveFocus()
  })

  it("filters the rows by every word typed, the highlight back on the first match", () => {
    const { search } = renderPicker()

    fireEvent.change(search, { target: { value: "E" } })
    fireEvent.keyDown(search, { key: "End" })
    expect(highlighted()).toBe("Leek")
    fireEvent.change(search, { target: { value: "e a" } })

    expect(screen.getAllByRole("option").map((item) => item.textContent)).toEqual(["Apple", "Pear"])
    expect(screen.queryByRole("group", { name: "Vegetables" })).not.toBeInTheDocument()
    expect(highlighted()).toBe("Apple")
  })

  it("says No match when nothing matches and picks nothing with Enter", () => {
    const { search, onPick } = renderPicker()

    fireEvent.change(search, { target: { value: "kiwi" } })
    fireEvent.keyDown(search, { key: "Enter" })

    expect(screen.queryByRole("option")).not.toBeInTheDocument()
    expect(screen.getByText("No match")).toBeInTheDocument()
    expect(search).toHaveAttribute("aria-expanded", "false")
    expect(onPick).not.toHaveBeenCalled()
  })

  it("keeps one highlight between the pointer and the keys, a pointer resting while the list scrolls moves nothing", () => {
    const { search, onPick } = renderPicker()

    fireEvent.mouseMove(option("Leek"), { clientX: 5, clientY: 40 })
    expect(highlighted()).toBe("Leek")
    fireEvent.keyDown(search, { key: "ArrowUp" })
    expect(highlighted()).toBe("Pear")
    fireEvent.mouseMove(option("Apple"), { clientX: 5, clientY: 40 })
    expect(highlighted()).toBe("Pear")
    fireEvent.click(option("Apple"))

    expect(onPick).toHaveBeenCalledExactlyOnceWith("Apple")
  })

  it("closes with Escape, focus back on the trigger, and opens again unfiltered", async () => {
    const { trigger, search, onPick } = renderPicker()

    fireEvent.change(search, { target: { value: "leek" } })
    fireEvent.keyDown(search, { key: "Escape" })
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    expect(trigger).toHaveFocus()
    fireEvent.click(trigger)

    expect(onPick).not.toHaveBeenCalled()
    expect(screen.getByRole("combobox", { name: "Search produce" })).toHaveValue("")
    expect(screen.getAllByRole("option")).toHaveLength(3)
    expect(highlighted()).toBe("Apple")
  })

  it("stays open with the search field focused when the pick could not be applied", () => {
    const { search } = renderPicker({ onPick: () => false })

    fireEvent.keyDown(search, { key: "Enter" })

    expect(screen.getByRole("dialog", { name: "Produce" })).toBeInTheDocument()
    expect(search).toHaveFocus()
  })

  it("previews the highlighted row in a floating window outside the list from the moment it opens, until it closes", async () => {
    const { search } = renderPicker({ renderPreview: (item) => `Preview of ${item}` })

    expect(card()).toHaveTextContent("Preview of Apple")
    expect(screen.getByRole("dialog")).not.toContainElement(card() as HTMLElement)
    fireEvent.keyDown(search, { key: "ArrowDown" })
    expect(card()).toHaveTextContent("Preview of Pear")
    fireEvent.change(search, { target: { value: "kiwi" } })
    expect(card()).toBeNull()
    fireEvent.change(search, { target: { value: "" } })
    fireEvent.keyDown(search, { key: "Escape" })

    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    expect(card()).toBeNull()
  })

  it("shows no window for a row without a preview", () => {
    const { search } = renderPicker({ renderPreview: (item) => item === "Pear" ? null : `Preview of ${item}` })

    expect(card()).toHaveTextContent("Preview of Apple")
    fireEvent.keyDown(search, { key: "ArrowDown" })

    expect(card()).toBeNull()
  })
})

describe("pickerLayout", () => {
  const landscape = { width: 1400, height: 850 }

  it.each([
    ["the window's own aspect", landscape.width / landscape.height],
    ["16:9", 16 / 9],
    ["a tall window's aspect", 0.8],
  ])("centers the preview exactly on the window at %s, the finder inside its top left corner", (_name, aspect) => {
    const { preview, finder } = pickerLayout(landscape, aspect)
    expect(preview.left + preview.width / 2).toBeCloseTo(landscape.width / 2)
    expect(preview.top + preview.height / 2).toBeCloseTo(landscape.height / 2)
    expect(preview.width / preview.height).toBeCloseTo(aspect)
    expect(preview.top).toBeGreaterThanOrEqual(44)
    expect(finder.left).toBeGreaterThan(preview.left)
    expect(finder.top).toBeGreaterThan(preview.top)
    expect(finder.top + finder.maxHeight).toBeLessThan(preview.top + preview.height)
  })
})
