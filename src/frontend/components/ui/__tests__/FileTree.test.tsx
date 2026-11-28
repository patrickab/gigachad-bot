import { afterEach, describe, expect, it } from "vitest"
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react"
import { FileTree, type FileTreeItem } from "@/components/ui/file-tree/FileTree"

const controller = { isExpanded: () => true, toggleBranch: () => {}, expand: () => {}, set: () => {} }

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

  it("wraps a row's actions in a div and gives an open section no delete button", () => {
    const project: FileTreeItem = {
      id: "p", label: "project", type: "folder",
      sections: [{ id: "chats", label: "chats", type: "folder", onDelete: () => {}, children: [
        { id: "c", label: "chat", type: "element", onDelete: () => {} },
      ] }],
    }
    render(<FileTree items={[project]} controller={controller} label="Projects" />)

    expect(screen.getByRole("button", { name: "Delete chat" }).parentElement?.tagName).toBe("DIV")
    expect(screen.queryByRole("button", { name: "Delete chats" })).toBeNull()
  })
})
