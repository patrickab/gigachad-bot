import { beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"

const api = vi.hoisted(() => ({ readMemory: vi.fn<(memoryStore: string, path: string) => Promise<unknown>>() }))

vi.mock("@/lib/backend/api", () => api)
vi.mock("@/components/content/LaTeXMarkdown", () => ({
  LaTeXMarkdown: ({ content }: { content: string }) => <output data-testid="rendered-markdown">{content}</output>,
}))

import { TagGroupDescription, WaveTree, waveStageLabel } from "@/features/factory/WaveTree"
import { treeOfWaves } from "@/features/factory/waveTree"
import type { FactoryTask, FactoryWave, FactoryWaveReview } from "@/lib/domain/factory"
import { factoryTask, factoryWave, waveTask } from "./factoryTask"

const PLAN = "gigachad-bot/factory-protocol"
const MISSING = Object.assign(new Error("not found"), { status: 404, code: "not_found" })

function renderTree(waves: FactoryWave[], tasks: FactoryTask[] = [], onOpenSession = vi.fn<(sessionId: string) => void>()) {
  return render(<WaveTree root={treeOfWaves(waves)} tasks={tasks} renderTask={(task) => <button type="button">{task.title}</button>} onOpenSession={onOpenSession} />)
}

function openWave(name: string) {
  fireEvent.click(screen.getByRole("button", { name: new RegExp(`^${name}`) }))
}

const USAGE = { input: 1200, output: 300, cost: 0.05 }

function review(fields: Partial<FactoryWaveReview> = {}): FactoryWaveReview {
  return { round: 1, at: "2026-01-01T01:00:00Z", sessionId: null, model: "claude-opus", verdict: "approved", status: "ok", findings: [], usage: USAGE, wallMs: 65_000, ...fields }
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
    expect(wave).not.toHaveTextContent("Board shows waves")
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

  it("shows a wave's reviews newest first: a rejected one with its findings open, an approved one", () => {
    renderTree([factoryWave("a", { tag: "w1", reviews: [
      review({ round: 1, verdict: "rejected", model: "gpt-5", findings: [
        { severity: "blocker", file: "src/app.ts", line: 12, text: "Leaks the token" },
        { severity: "minor", file: "README.md", line: null, text: "Typo" },
      ] }),
      review({ round: 2, at: "2026-01-01T02:00:00Z" }),
    ] })])
    openWave("w1")

    const rows = within(screen.getByRole("list", { name: "Outcome" })).getAllByRole("listitem").filter((item) => item.textContent?.startsWith("review"))
    expect(rows).toHaveLength(2)
    expect(rows[0]).toHaveTextContent("claude-opus")
    expect(rows[0]).toHaveTextContent("1.2k in / 300 out · $0.050 · ")
    expect(rows[0]).toHaveTextContent("approved")
    expect(rows[1]).toHaveTextContent("gpt-5")
    expect(rows[1]).toHaveTextContent("rejected")
    expect(rows[1]).toHaveTextContent("2 findings")
    expect(rows[1]).toHaveTextContent("src/app.ts:12")
    expect(rows[1]).toHaveTextContent("blocker")
    expect(rows[1]).toHaveTextContent("Leaks the token")
    expect(within(rows[1]).getByText("README.md")).toBeTruthy()
  })

  it("names a failed review by its status", () => {
    renderTree([factoryWave("a", { tag: "w1", reviews: [review({ verdict: "failed", status: "timeout", usage: null })] })])
    openWave("w1")
    expect(screen.getByRole("list", { name: "Outcome" })).toHaveTextContent("timed out")
  })

  it("shows the merged commit range short", () => {
    renderTree([factoryWave("a", { tag: "w1", stage: "merged", merge: { at: "2026-01-01T03:00:00Z", from: "0123456789abcdef", to: "fedcba9876543210", queueMs: 0 } })])
    openWave("w1")
    expect(screen.getByLabelText("Merged range")).toHaveTextContent("01234567..fedcba98")
  })

  it("shows a failed gate run with its log, and no passed one", () => {
    renderTree([factoryWave("a", { tag: "w1", gates: [
      { at: "2026-01-01T01:00:00Z", phase: "review", passed: true, wallMs: 1000, log: "all green" },
      { at: "2026-01-01T02:00:00Z", phase: "merge", passed: false, wallMs: 2000, log: "1 test failed" },
    ] })])
    openWave("w1")
    const outcome = screen.getByRole("list", { name: "Outcome" })
    expect(outcome).toHaveTextContent("merge gate")
    expect(outcome).toHaveTextContent("1 test failed")
    expect(outcome).not.toHaveTextContent("review gate")
    expect(outcome).not.toHaveTextContent("all green")
  })

  it("shows nothing of the outcome while only passed gates ran", () => {
    renderTree([factoryWave("a", { tag: "w1", gates: [{ at: "2026-01-01T01:00:00Z", phase: "review", passed: true, wallMs: 1000, log: null }] })])
    openWave("w1")
    expect(screen.queryByRole("list", { name: "Outcome" })).toBeNull()
  })

  it("opens the supervisor's and the reviewer's Pi sessions by their ids", () => {
    const onOpenSession = vi.fn<(sessionId: string) => void>()
    renderTree([factoryWave("a", { tag: "w1", title: "Board shows waves", origin: "supervisor-1", reviews: [review({ sessionId: "reviewer-1" })] })], [], onOpenSession)
    openWave("w1")

    expect(screen.getByText("Board shows waves")).toBeTruthy()
    fireEvent.click(screen.getByRole("button", { name: "Open supervisor session" }))
    fireEvent.click(screen.getByRole("button", { name: "Open reviewer session" }))
    expect(onOpenSession.mock.calls).toEqual([["supervisor-1"], ["reviewer-1"]])
  })

  it("has no session links without the sessions' ids", () => {
    renderTree([factoryWave("a", { tag: "w1", origin: null, reviews: [review({ sessionId: null })] })])
    openWave("w1")
    expect(screen.queryByRole("button", { name: "Open supervisor session" })).toBeNull()
    expect(screen.queryByRole("button", { name: "Open reviewer session" })).toBeNull()
  })

  it("links the remote request", () => {
    renderTree([factoryWave("a", { tag: "w1", remote: { number: 42, url: "https://example.com/pr/42", pushed: null, error: null } })])
    openWave("w1")
    expect(screen.getByRole("link", { name: "#42" })).toHaveAttribute("href", "https://example.com/pr/42")
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
