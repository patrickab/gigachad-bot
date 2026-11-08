import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react"
import { Archive, Upload } from "lucide-react"
import { FileTree, type FileTreeItem } from "@/components/ui/file-tree/FileTree"

const controller = { isExpanded: () => true, toggleBranch: () => {}, expand: () => {} }

function folder(count: number, { activeIndex }: { activeIndex?: number } = {}): FileTreeItem {
  return {
    id: "code",
    label: "code",
    type: "folder",
    childLimit: 5,
    children: Array.from({ length: count }, (_, index) => ({ id: `s${index}`, label: `Session ${index}`, type: "element", isActive: index === activeIndex })),
  }
}

const shownLabels = () => within(screen.getByRole("group", { name: "code contents" })).queryAllByRole("button")
  .map((button) => button.textContent)

describe("FileTree childLimit", () => {
  afterEach(cleanup)

  it("shows every child without a toggle up to the limit", () => {
    render(<FileTree items={[folder(5)]} controller={controller} label="Sessions" />)

    expect(shownLabels()).toEqual(["Session 0", "Session 1", "Session 2", "Session 3", "Session 4"])
  })

  it("caps longer lists and toggles the rest", () => {
    render(<FileTree items={[folder(7)]} controller={controller} label="Sessions" />)

    expect(shownLabels()).toEqual(["Session 0", "Session 1", "Session 2", "Session 3", "Session 4", "Show 2 more"])
    fireEvent.click(screen.getByRole("button", { name: "Show 2 more" }))
    expect(shownLabels()).toEqual(["Session 0", "Session 1", "Session 2", "Session 3", "Session 4", "Session 5", "Session 6", "Show less"])
    fireEvent.click(screen.getByRole("button", { name: "Show less" }))
    expect(shownLabels()).toHaveLength(6)
  })

  it("keeps the active child visible past the limit", () => {
    render(<FileTree items={[folder(8, { activeIndex: 6 })]} controller={controller} label="Sessions" />)

    expect(shownLabels()).toEqual(["Session 0", "Session 1", "Session 2", "Session 3", "Session 4", "Session 6", "Show 2 more"])
  })
})

describe("FileTree row actions", () => {
  afterEach(cleanup)

  it("keeps the first action and delete inline and moves the rest into the row's menu, danger last, with short visible labels", () => {
    const archive = vi.fn()
    const purge = vi.fn()
    const remove = vi.fn()
    const item: FileTreeItem = {
      id: "p", label: "Proj", type: "root", onDelete: remove, children: [],
      actions: [
        { label: "Upload to Proj", icon: Upload, onClick: vi.fn() },
        { label: "Purge Proj", menuLabel: "Purge", icon: Archive, onClick: purge, danger: true },
        { label: "Archive Proj", menuLabel: "Archive", icon: Archive, onClick: archive },
      ],
    }
    render(<FileTree items={[item]} controller={controller} label="Projects" />)

    const inline = screen.getAllByRole("button").map((button) => button.getAttribute("aria-label"))
    expect(inline.slice(-3)).toEqual(["Upload to Proj", "More actions for Proj", "Delete Proj"])
    fireEvent.click(screen.getByRole("button", { name: "More actions for Proj" }))
    const menu = within(screen.getByRole("menu", { name: "Proj actions" }))
    expect(menu.getAllByRole("menuitem").map((entry) => entry.textContent)).toEqual(["Archive", "Purge"])
    expect(menu.queryByRole("menuitem", { name: "Delete Proj" })).not.toBeInTheDocument()
    expect(menu.getByRole("menuitem", { name: "Archive Proj" })).toHaveFocus()

    fireEvent.keyDown(menu.getByRole("menuitem", { name: "Archive Proj" }), { key: "ArrowDown" })
    expect(menu.getByRole("menuitem", { name: "Purge Proj" })).toHaveFocus()
    fireEvent.click(menu.getByRole("menuitem", { name: "Purge Proj" }))
    expect(purge).toHaveBeenCalledOnce()
    expect(screen.queryByRole("menu")).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Delete Proj" }))
    expect(remove).toHaveBeenCalledOnce()
  })

  describe("menu placement", () => {
    const style = document.createElement("style")
    style.textContent = "[role=menu] { position: absolute; }"
    // The row's menu button is 20px high, its 100px menu sits 4px below it; the scroll container ends at 400px.
    let anchorTop = 0

    beforeEach(() => {
      document.head.append(style)
      vi.spyOn(Element.prototype, "getBoundingClientRect").mockImplementation(function (this: Element) {
        return (this.hasAttribute("data-menu-popover") ? { top: anchorTop, bottom: anchorTop + 20 } : { top: 0, bottom: 400 }) as DOMRect
      })
      vi.spyOn(HTMLElement.prototype, "offsetHeight", "get").mockImplementation(function (this: HTMLElement) {
        return this.getAttribute("role") === "menu" ? 100 : 20
      })
      vi.spyOn(HTMLElement.prototype, "offsetTop", "get").mockReturnValue(24)
    })
    afterEach(() => {
      style.remove()
      vi.restoreAllMocks()
    })

    const openMenu = () => {
      const actions = ["Upload", "Archive"].map((name) => ({ label: `${name} Proj`, icon: Archive, onClick: vi.fn() }))
      const item: FileTreeItem = { id: "p", label: "Proj", type: "root", actions, children: [] }
      render(<FileTree items={[item]} controller={controller} label="Projects" />)
      fireEvent.click(screen.getByRole("button", { name: "More actions for Proj" }))
      return screen.getByRole("menu", { name: "Proj actions" })
    }

    it("opens upward, keeping its gap, when the scroll container has no room below", () => {
      anchorTop = 300
      expect(openMenu()).toHaveStyle({ top: "auto", bottom: "calc(100% + 4px)" })
    })

    it("opens downward when it fits below", () => {
      anchorTop = 100
      expect(openMenu().style.bottom).toBe("")
    })
  })

  it("shows an expanded branch without children as empty", () => {
    render(<FileTree items={[{ id: "c", label: "chats", type: "folder", children: [] }]} controller={controller} label="Projects" />)

    expect(within(screen.getByRole("group", { name: "chats contents" })).getByText("Empty")).toBeInTheDocument()
  })
})
