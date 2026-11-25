import { beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"

const api = vi.hoisted(() => ({ readMemory: vi.fn<(memoryStore: string, path: string) => Promise<unknown>>() }))

vi.mock("@/lib/backend/api", () => api)
vi.mock("@/components/content/LaTeXMarkdown", () => ({
  LaTeXMarkdown: ({ content }: { content: string }) => <output data-testid="rendered-markdown">{content}</output>,
}))

import { TagGroupDescription, WaveContainers } from "@/features/factory/WaveTree"
import { waveStageLabel } from "@/lib/domain/factory"
import { treeOfWaves } from "@/features/factory/waveTree"
import type { FactoryTask, FactoryWave, FactoryWaveReview } from "@/lib/domain/factory"
import { factoryTask, factoryWave } from "./factoryTask"

const PLAN = "gigachad-bot/factory-protocol"
const MISSING = Object.assign(new Error("not found"), { status: 404, code: "not_found" })

/** The waves' containers, open unless `defaultOpen` is false, `done` their done tasks. */
function renderTree(waves: FactoryWave[], done: FactoryTask[] = [], onOpenSession = vi.fn<(sessionId: string) => void>(), defaultOpen = true) {
  const doneTasks = new Map<string, FactoryTask[]>()
  for (const task of done) doneTasks.set(task.wave!, [...(doneTasks.get(task.wave!) ?? []), task])
  return render(
    <WaveContainers
      group={treeOfWaves(waves)}
      doneTasks={doneTasks}
      renderTask={(task) => <button type="button">{task.title}</button>}
      onOpenSession={onOpenSession}
      defaultOpen={defaultOpen}
    />,
  )
}

/** The header of the container named `name`. */
function header(name: string) {
  return within(screen.getByRole("group", { name })).getAllByRole("button")[0]
}

const USAGE = { input: 1200, output: 300, cost: 0.05 }

function review(fields: Partial<FactoryWaveReview> = {}): FactoryWaveReview {
  return { round: 1, at: "2026-01-01T01:00:00Z", sessionId: null, model: "claude-opus", verdict: "approved", status: "ok", findings: [], usage: USAGE, wallMs: 65_000, ...fields }
}

beforeEach(() => {
  api.readMemory.mockReset()
  api.readMemory.mockRejectedValue(MISSING)
})

describe("WaveContainers", () => {
  it("orders a wave's header: title and count on one row, description toggle, then stage and usage", async () => {
    api.readMemory.mockResolvedValue({ kind: "document", text: "# Wave one" })
    renderTree([factoryWave("a", { tag: "w1", title: "Wave one", plan: PLAN, counts: { total: 5, done: 1 }, usage: { input: 50_000, output: 12_000, cost: 3.5, cacheRead: 5000 } })])
    const toggle = await screen.findByRole("button", { name: "Description" })
    const title = header("Wave one")
    expect(title.textContent).toMatch(/^Wave one1\/5$/)
    const stage = screen.getByLabelText("Tokens and cost")
    expect(stage).toHaveTextContent("↑55k ↓12k · $3.50")
    expect(title.compareDocumentPosition(toggle) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(toggle.compareDocumentPosition(stage) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(toggle).toHaveAttribute("aria-expanded", "false")
  })

  it("heads a wave with done/total, its title, stage and usage, and a milestone with merged/waves and summed usage", () => {
    renderTree([
      factoryWave("a", { tag: "plugin/m1/w1", title: "Board shows waves", stage: "reviewing", counts: { total: 8, done: 5 }, usage: { input: 1500, output: 500, cost: 0.42 } }),
      factoryWave("b", { tag: "plugin/m1/w2", title: "Second", stage: "merged", counts: { total: 2, done: 2 }, usage: { input: 0, output: 1000, cost: 0.08 } }),
    ])
    const track = header("plugin")
    expect(within(track).getByLabelText("Progress")).toHaveTextContent("1/2")
    expect(within(screen.getByRole("group", { name: "plugin" })).getAllByLabelText("Tokens and cost")[0]).toHaveTextContent("↑1.5k ↓1.5k · $0.50")
    expect(within(header("m1")).getByLabelText("Progress")).toHaveTextContent("1/2")

    const wave = header("Board shows waves")
    expect(wave).toHaveTextContent("Board shows waves")
    expect(within(wave).getByLabelText("Progress")).toHaveTextContent("5/8")
    const frame = within(screen.getByRole("group", { name: "Board shows waves" }))
    expect(frame.getByLabelText("Stage")).toHaveTextContent("reviewing")
    expect(frame.getByLabelText("Tokens and cost")).toHaveTextContent("↑1.5k ↓500 · $0.42")
  })

  it("draws a wave alone in its track without the track's frame, two waves with it", () => {
    const { unmount } = renderTree([factoryWave("a", { tag: "plugin/w1", title: "Only wave" })])
    expect(screen.getByRole("group", { name: "Only wave" })).toBeTruthy()
    expect(screen.queryByRole("group", { name: "plugin" })).toBeNull()
    unmount()
    renderTree([factoryWave("a", { tag: "plugin/w1", title: "First" }), factoryWave("b", { tag: "plugin/w2", title: "Second" })])
    expect(screen.getByRole("group", { name: "plugin" })).toBeTruthy()
  })

  it("names an interrupted wave so, and a blocked one's reason", () => {
    expect(waveStageLabel({ stage: "building", reason: null, interrupted: true })).toBe("interrupted")
    expect(waveStageLabel({ stage: "blocked", reason: "gates", interrupted: false })).toBe("blocked · gates")
    expect(waveStageLabel({ stage: "waiting_approve", reason: null, interrupted: false })).toBe("ready for review")
  })

  it("holds a wave's done tasks as cards, no pending specs, and closes from its header", () => {
    renderTree([factoryWave("a", { tag: "w1", title: "Wave one", counts: { total: 3, done: 1 } })], [factoryTask("t1", { title: "Task done", state: "done", wave: "a" })])
    const wave = within(screen.getByRole("group", { name: "Wave one" }))
    expect(wave.getByRole("button", { name: "Task done" })).toBeTruthy()
    expect(screen.queryByRole("list", { name: "Pending" })).toBeNull()

    fireEvent.click(header("Wave one"))
    expect(wave.queryByRole("button", { name: "Task done" })).toBeNull()
  })

  it("starts closed when asked", () => {
    renderTree([factoryWave("a", { tag: "w1", title: "Wave one", stage: "merged" })], [factoryTask("t1", { title: "Task done", state: "done", wave: "a" })], vi.fn(), false)
    expect(header("Wave one")).toHaveAttribute("aria-expanded", "false")
    expect(screen.queryByRole("button", { name: "Task done" })).toBeNull()
  })

  it("reads no wave's description while its toggle is closed, a closed wave's neither", () => {
    renderTree([factoryWave("a", { tag: "w1", title: "Wave one", stage: "merged", plan: PLAN })], [], vi.fn(), false)
    expect(screen.getByRole("button", { name: "Description" })).toHaveAttribute("aria-expanded", "false")
    expect(api.readMemory).not.toHaveBeenCalled()
  })

  it("is a disclosure: its header controls the content below the description and meta", () => {
    renderTree([factoryWave("a", { tag: "w1", title: "Wave one", stage: "merged" })], [factoryTask("t1", { title: "Task done", state: "done", wave: "a" })])
    const title = header("Wave one")
    const content = document.getElementById(title.getAttribute("aria-controls") ?? "")
    expect(content).toContainElement(screen.getByRole("button", { name: "Task done" }))
    expect(content).not.toContainElement(screen.getByLabelText("Stage"))
  })

  it("titles a track by the first line of its description, falling back to its name", async () => {
    api.readMemory.mockImplementation(async (_store, path) => {
      if (path === "plan/plugin.md") return { kind: "document", text: "# Plugin track\n\nBody" }
      throw MISSING
    })
    renderTree([factoryWave("a", { tag: "plugin/m1/w1", title: "Wave", plan: PLAN }), factoryWave("b", { tag: "plugin/m1/w2", title: "Wave two", plan: PLAN })])
    expect(await screen.findByRole("group", { name: "Plugin track" })).toBeTruthy()
    expect(screen.getByRole("group", { name: "m1" })).toBeTruthy()
    expect(api.readMemory).toHaveBeenCalledWith(PLAN, "plan/plugin/m1.md")
  })

  it("shows a wave's reviews newest first: a rejected one with its findings open, an approved one", () => {
    renderTree([factoryWave("a", { tag: "w1", reviews: [
      review({ round: 1, verdict: "rejected", model: "gpt-5", findings: [
        { severity: "blocker", file: "src/app.ts", line: 12, text: "Leaks the token" },
        { severity: "minor", file: "README.md", line: null, text: "Typo" },
      ] }),
      review({ round: 2, at: "2026-01-01T02:00:00Z" }),
    ] })])

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
    expect(screen.getByRole("list", { name: "Outcome" })).toHaveTextContent("timed out")
  })

  it("shows the merged commit range short", () => {
    renderTree([factoryWave("a", { tag: "w1", stage: "merged", merge: { at: "2026-01-01T03:00:00Z", from: "0123456789abcdef", to: "fedcba9876543210", queueMs: 0 } })])
    expect(screen.getByLabelText("Merged range")).toHaveTextContent("01234567..fedcba98")
  })

  it("shows a failed gate run with its log, and no passed one", () => {
    renderTree([factoryWave("a", { tag: "w1", gates: [
      { at: "2026-01-01T01:00:00Z", phase: "review", passed: true, wallMs: 1000, log: "all green" },
      { at: "2026-01-01T02:00:00Z", phase: "merge", passed: false, wallMs: 2000, log: "1 test failed" },
    ] })])
    const outcome = screen.getByRole("list", { name: "Outcome" })
    expect(outcome).toHaveTextContent("merge gate")
    expect(outcome).toHaveTextContent("1 test failed")
    expect(outcome).not.toHaveTextContent("review gate")
    expect(outcome).not.toHaveTextContent("all green")
  })

  it("shows nothing of the outcome while only passed gates ran", () => {
    renderTree([factoryWave("a", { tag: "w1", gates: [{ at: "2026-01-01T01:00:00Z", phase: "review", passed: true, wallMs: 1000, log: null }] })])
    expect(screen.queryByRole("list", { name: "Outcome" })).toBeNull()
  })

  it("opens the supervisor's and the reviewer's Pi sessions by their ids", () => {
    const onOpenSession = vi.fn<(sessionId: string) => void>()
    renderTree([factoryWave("a", { tag: "w1", title: "Board shows waves", origin: "supervisor-1", reviews: [review({ sessionId: "reviewer-1" })] })], [], onOpenSession)

    expect(screen.getByText("Board shows waves")).toBeTruthy()
    fireEvent.click(screen.getByRole("button", { name: "Open supervisor session" }))
    fireEvent.click(screen.getByRole("button", { name: "Open reviewer session" }))
    expect(onOpenSession.mock.calls).toEqual([["supervisor-1"], ["reviewer-1"]])
  })

  it("has no session links without the sessions' ids", () => {
    renderTree([factoryWave("a", { tag: "w1", origin: null, reviews: [review({ sessionId: null })] })])
    expect(screen.queryByRole("button", { name: "Open supervisor session" })).toBeNull()
    expect(screen.queryByRole("button", { name: "Open reviewer session" })).toBeNull()
  })

  it("links the remote request", () => {
    renderTree([factoryWave("a", { tag: "w1", remote: { number: 42, url: "https://example.com/pr/42", pushed: null, error: null } })])
    expect(screen.getByRole("link", { name: "#42" })).toHaveAttribute("href", "https://example.com/pr/42")
  })

  it("reads a wave's description by its tag", async () => {
    api.readMemory.mockImplementation(async (_store, path) => {
      if (path === "plan/w1.md") return { kind: "document", text: "# Wave one" }
      throw MISSING
    })
    renderTree([factoryWave("a", { tag: "w1", plan: PLAN })])
    fireEvent.click(await screen.findByRole("button", { name: "Description" }))
    expect(await screen.findByTestId("rendered-markdown")).toHaveTextContent("# Wave one")
  })
})

describe("TagGroupDescription", () => {
  it("reads its document only once opened", async () => {
    api.readMemory.mockResolvedValue({ kind: "document", text: "# Wave one" })
    render(<TagGroupDescription plan={PLAN} path="plugin/m1" />)
    const toggle = screen.getByRole("button", { name: "Description" })
    expect(api.readMemory).not.toHaveBeenCalled()
    fireEvent.click(toggle)
    expect(await screen.findByTestId("rendered-markdown")).toHaveTextContent("# Wave one")
    expect(api.readMemory).toHaveBeenCalledExactlyOnceWith(PLAN, "plan/plugin/m1.md")
  })

  it("opens to nothing when the document or memory_store is missing", async () => {
    render(<TagGroupDescription plan={PLAN} path="plugin/m1" />)
    fireEvent.click(screen.getByRole("button", { name: "Description" }))
    await waitFor(() => expect(api.readMemory).toHaveBeenCalledWith(PLAN, "plan/plugin/m1.md"))
    expect(screen.queryByTestId("rendered-markdown")).toBeNull()
  })

  it("renders nothing inside for an empty document", async () => {
    api.readMemory.mockResolvedValue({ kind: "document", text: "  \n" })
    render(<TagGroupDescription plan={PLAN} path="plugin" />)
    fireEvent.click(screen.getByRole("button", { name: "Description" }))
    await waitFor(() => expect(api.readMemory).toHaveBeenCalledOnce())
    expect(screen.queryByTestId("rendered-markdown")).toBeNull()
  })
})
