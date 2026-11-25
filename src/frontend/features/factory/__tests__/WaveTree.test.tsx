import { beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"

const api = vi.hoisted(() => ({ readMemory: vi.fn<(memoryStore: string, path: string) => Promise<unknown>>() }))

vi.mock("@/lib/backend/api", () => api)
vi.mock("@/components/content/LaTeXMarkdown", () => ({
  LaTeXMarkdown: ({ content }: { content: string }) => <output data-testid="rendered-markdown">{content}</output>,
}))

import { TagGroupDescription, WaveTree, waveStageLabel } from "@/features/factory/WaveTree"
import { treeOfWaves } from "@/features/factory/waveTree"
import type { FactoryTask, FactoryWave } from "@/lib/domain/factory"
import { factoryTask, factoryWave, waveTask } from "./factoryTask"

const PLAN = "gigachad-bot/factory-protocol"
const MISSING = Object.assign(new Error("not found"), { status: 404, code: "not_found" })

function renderTree(waves: FactoryWave[], tasks: FactoryTask[] = []) {
  return render(<WaveTree root={treeOfWaves(waves)} tasks={tasks} renderTask={(task) => <button type="button">{task.title}</button>} />)
}

beforeEach(() => {
  api.readMemory.mockReset()
  api.readMemory.mockRejectedValue(MISSING)
})

describe("WaveTree", () => {
  it("shows a track's sums, and a wave row's progress, stage and usage", () => {
    renderTree([
      factoryWave("a", { tag: "plugin/m1/w1", title: "Board shows waves", stage: "reviewing", counts: { total: 8, done: 3 }, usage: { input: 1500, output: 500, cost: 0.42 } }),
      factoryWave("b", { tag: "plugin/m1/w2", counts: { total: 2, done: 2 }, usage: { input: 0, output: 1000, cost: 0.08 } }),
    ])
    const track = screen.getByRole("button", { name: /^plugin/ })
    expect(within(track).getByLabelText("Progress")).toHaveTextContent("5/10")
    expect(within(track).getByLabelText("Tokens and cost")).toHaveTextContent("3.0k tok · $0.50")
    expect(within(track).queryByLabelText("Stage")).toBeNull()

    fireEvent.click(track)
    fireEvent.click(screen.getByRole("button", { name: /^m1/ }))
    const wave = screen.getByRole("button", { name: /^w1/ })
    expect(wave).toHaveTextContent("Board shows waves")
    expect(within(wave).getByLabelText("Progress")).toHaveTextContent("3/8")
    expect(within(wave).getByLabelText("Stage")).toHaveTextContent("reviewing")
    expect(within(wave).getByLabelText("Tokens and cost")).toHaveTextContent("2.0k tok · $0.42")
  })

  it("names an interrupted wave so, and a blocked one's reason", () => {
    expect(waveStageLabel({ stage: "building", reason: null, interrupted: true })).toBe("interrupted")
    expect(waveStageLabel({ stage: "blocked", reason: "gates", interrupted: false })).toBe("blocked · gates")
    expect(waveStageLabel({ stage: "waiting_approve", reason: null, interrupted: false })).toBe("ready for review")
  })

  it("opens a wave to its tasks as cards and its specs not started yet as quiet rows", () => {
    const started = factoryTask("t1", { title: "Task started", wave: "a" })
    renderTree(
      [factoryWave("a", { tag: "w1", tasks: [waveTask("one", { taskId: "t1", state: "running" }), waveTask("two", { title: "Later work" })] })],
      [started, factoryTask("t2", { title: "Task elsewhere" })],
    )
    fireEvent.click(screen.getByRole("button", { name: /^w1/ }))

    expect(screen.getByRole("button", { name: "Task started" })).toBeTruthy()
    expect(screen.queryByRole("button", { name: "Task elsewhere" })).toBeNull()
    const pending = within(screen.getByRole("list", { name: "Pending" }))
    expect(pending.getAllByRole("listitem").map((item) => item.textContent)).toEqual(["Later work"])
  })

  it("reads a wave's description by its tag when it opens", async () => {
    api.readMemory.mockImplementation(async (_store, path) => {
      if (path === "plan/p/w1.md") return { kind: "document", text: "# Wave one" }
      throw MISSING
    })
    renderTree([factoryWave("a", { tag: "p/w1", plan: PLAN })])
    expect(api.readMemory).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole("button", { name: /^p/ }))
    fireEvent.click(screen.getByRole("button", { name: /^w1/ }))

    fireEvent.click(await screen.findByRole("button", { name: "Description" }))
    expect(api.readMemory).toHaveBeenCalledWith(PLAN, "plan/p.md")
    expect(screen.getByTestId("rendered-markdown")).toHaveTextContent("# Wave one")
  })
})

describe("TagGroupDescription", () => {
  it("renders nothing when the document or memory_store is missing", async () => {
    const { container } = render(<TagGroupDescription plan={PLAN} path="plugin/m1" />)
    await waitFor(() => expect(api.readMemory).toHaveBeenCalledWith(PLAN, "plan/plugin/m1.md"))
    expect(container).toBeEmptyDOMElement()
  })

  it("renders nothing for an empty document", async () => {
    api.readMemory.mockResolvedValue({ kind: "document", text: "  \n" })
    const { container } = render(<TagGroupDescription plan={PLAN} path="plugin" />)
    await waitFor(() => expect(api.readMemory).toHaveBeenCalledOnce())
    expect(container).toBeEmptyDOMElement()
  })
})
