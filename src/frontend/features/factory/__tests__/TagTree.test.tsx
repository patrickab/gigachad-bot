import { beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"

const api = vi.hoisted(() => ({ readMemory: vi.fn<(memoryStore: string, path: string) => Promise<unknown>>() }))

vi.mock("@/lib/backend/api", () => api)
vi.mock("@/components/content/LaTeXMarkdown", () => ({
  LaTeXMarkdown: ({ content }: { content: string }) => <output data-testid="rendered-markdown">{content}</output>,
}))

import { TagGroupDescription, TagTree } from "@/features/factory/TagTree"
import { groupByTag } from "@/features/factory/tagGroups"
import { factoryTask } from "./factoryTask"

const PLAN = "gigachad-bot/factory-protocol"

function renderTree() {
  const tasks = [
    factoryTask("t1", { tag: "plugin/m1/w1", title: "Task one", plan: PLAN }),
    factoryTask("t2", { tag: "plugin/m1/w1", title: "Task two", plan: PLAN, state: "done", running: false }),
  ]
  return render(<TagTree groups={groupByTag(tasks).groups} renderTask={(task) => <button type="button">{task.title}</button>} />)
}

const MISSING = Object.assign(new Error("not found"), { status: 404, code: "not_found" })

beforeEach(() => {
  api.readMemory.mockReset()
  api.readMemory.mockRejectedValue(MISSING)
})

describe("TagTree", () => {
  it("opens group by group down to the wave, whose tasks show as cards", async () => {
    renderTree()
    const plugin = screen.getByRole("button", { name: /^plugin/ })
    expect(plugin).toHaveAttribute("aria-expanded", "false")
    expect(plugin).toHaveTextContent("plugin2")
    expect(screen.queryByText("Task one")).toBeNull()
    expect(api.readMemory).not.toHaveBeenCalled()

    fireEvent.click(plugin)
    expect(plugin).toHaveAttribute("aria-expanded", "true")
    fireEvent.click(screen.getByRole("button", { name: /^m1/ }))
    const wave = screen.getByRole("button", { name: /^w1/ })
    expect(wave).toHaveTextContent("1 working · 1 done")
    fireEvent.click(wave)

    expect(screen.getByRole("button", { name: "Task one" })).toBeTruthy()
    expect(screen.getByRole("button", { name: "Task two" })).toBeTruthy()
    await waitFor(() => expect(api.readMemory).toHaveBeenCalledTimes(3))
    expect(screen.queryByRole("button", { name: "Description" })).toBeNull()
  })

  it("reads a group's description when the group opens and shows its row only for an existing document", async () => {
    api.readMemory.mockImplementation(async (_store, path) => {
      if (path === "plan/plugin.md") return { kind: "document", text: "# Plugin track" }
      throw MISSING
    })
    renderTree()
    fireEvent.click(screen.getByRole("button", { name: /^plugin/ }))

    const description = await screen.findByRole("button", { name: "Description" })
    expect(api.readMemory).toHaveBeenCalledWith(PLAN, "plan/plugin.md")
    expect(screen.queryByTestId("rendered-markdown")).toBeNull()
    fireEvent.click(description)
    expect(screen.getByTestId("rendered-markdown")).toHaveTextContent("# Plugin track")

    fireEvent.click(screen.getByRole("button", { name: /^m1/ }))
    await waitFor(() => expect(api.readMemory).toHaveBeenCalledWith(PLAN, "plan/plugin/m1.md"))
    expect(screen.getAllByRole("button", { name: "Description" })).toHaveLength(1)
  })
})

describe("TagGroupDescription", () => {
  it("renders nothing when the document or memory_store is missing", async () => {
    const { container } = render(<TagGroupDescription plan={PLAN} path="plugin/m1" />)

    await waitFor(() => expect(api.readMemory).toHaveBeenCalledWith(PLAN, "plan/plugin/m1.md"))
    expect(container).toBeEmptyDOMElement()
    expect(screen.queryByRole("alert")).toBeNull()
  })

  it("renders nothing for an empty document", async () => {
    api.readMemory.mockResolvedValue({ kind: "document", text: "  \n" })
    const { container } = render(<TagGroupDescription plan={PLAN} path="plugin" />)

    await waitFor(() => expect(api.readMemory).toHaveBeenCalledOnce())
    expect(container).toBeEmptyDOMElement()
  })
})
