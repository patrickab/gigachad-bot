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

beforeEach(() => {
  api.readMemory.mockReset()
})

describe("TagTree", () => {
  it("opens group by group down to the wave, whose tasks show as cards", () => {
    renderTree()
    const plugin = screen.getByRole("button", { name: /^plugin/ })
    expect(plugin).toHaveAttribute("aria-expanded", "false")
    expect(plugin).toHaveTextContent("plugin2")
    expect(screen.queryByText("Task one")).toBeNull()

    fireEvent.click(plugin)
    expect(plugin).toHaveAttribute("aria-expanded", "true")
    fireEvent.click(screen.getByRole("button", { name: /^m1/ }))
    const wave = screen.getByRole("button", { name: /^w1/ })
    expect(wave).toHaveTextContent("1 working · 1 done")
    fireEvent.click(wave)

    expect(screen.getByRole("button", { name: "Task one" })).toBeTruthy()
    expect(screen.getByRole("button", { name: "Task two" })).toBeTruthy()
    expect(api.readMemory).not.toHaveBeenCalled()
  })

  it("reads a group's description only when asked and renders it as markdown", async () => {
    api.readMemory.mockResolvedValue({ kind: "document", text: "# Plugin track" })
    renderTree()
    fireEvent.click(screen.getByRole("button", { name: /^plugin/ }))

    fireEvent.click(screen.getByRole("button", { name: "Description" }))

    expect(await screen.findByTestId("rendered-markdown")).toHaveTextContent("# Plugin track")
    expect(api.readMemory).toHaveBeenCalledWith(PLAN, "plan/plugin.md")
  })
})

describe("TagGroupDescription", () => {
  it("renders nothing when the document or memory_store is missing", async () => {
    api.readMemory.mockRejectedValue(Object.assign(new Error("not found"), { status: 404, code: "not_found" }))
    const { container } = render(<TagGroupDescription plan={PLAN} path="plugin/m1" />)

    await waitFor(() => expect(api.readMemory).toHaveBeenCalledWith(PLAN, "plan/plugin/m1.md"))
    expect(container).toBeEmptyDOMElement()
    expect(screen.queryByRole("alert")).toBeNull()
  })
})
