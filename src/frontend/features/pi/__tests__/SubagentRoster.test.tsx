import { afterEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react"
import { SubagentRoster, useSubagentRoster } from "../SubagentRoster"
import type { SubagentRow } from "../subagents"
import type { ToolRun } from "../types"

afterEach(cleanup)

const rows: SubagentRow[] = [
  { id: "a", agent: "scout", task: "Map the repo", status: "running", tool: "grep", tokens: 12_300, cost: 0.05, durationMs: 4000 },
  { id: "b", agent: "worker", task: "Fix it", status: "done", output: "Fixed the bug" },
  { id: "c", agent: "bg", status: "failed", inspect: { asyncId: "run", childId: "c" } },
]

function renderRoster(overrides: Partial<Parameters<typeof SubagentRoster>[0]> = {}) {
  const props = { rows, onDismiss: vi.fn(), onClear: vi.fn(), inspect: vi.fn().mockResolvedValue({ requestId: "r", task: "Full task", finalOutput: "Background output" }), ...overrides }
  render(<SubagentRoster {...props} />)
  return props
}

describe("SubagentRoster", () => {
  it("renders nothing without children", () => {
    renderRoster({ rows: [] })
    expect(screen.queryByRole("region", { name: "Subagents" })).toBeNull()
  })

  it("shows a row per child with agent, task, tool, tokens, cost and a status icon", () => {
    renderRoster()
    expect(screen.getByRole("button", { name: /1 running · 2 done|3 subagents/ })).toBeTruthy()
    const scout = screen.getByRole("button", { name: /^Running scout/ })
    expect(scout.textContent).toContain("Map the repo")
    expect(scout.textContent).toContain("grep")
    expect(scout.textContent).toContain("12.3k tok")
    expect(scout.textContent).toContain("$0.05")
    expect(screen.getByLabelText("Running")).toBeTruthy()
    expect(screen.getByLabelText("Done")).toBeTruthy()
    expect(screen.getByLabelText("Failed")).toBeTruthy()
  })

  it("collapses to its summary line", () => {
    renderRoster()
    fireEvent.click(screen.getByRole("button", { name: /3 subagents/ }))
    expect(screen.queryByText("Map the repo")).toBeNull()
  })

  it("opens a foreground child's reported output without inspecting", () => {
    const { inspect } = renderRoster()
    fireEvent.click(screen.getByRole("button", { name: /^Done worker/ }))
    expect(screen.getByLabelText("worker output").textContent).toContain("Fixed the bug")
    expect(inspect).not.toHaveBeenCalled()
  })

  it("opens a background child's task and output through inspect", async () => {
    const { inspect } = renderRoster()
    fireEvent.click(screen.getByRole("button", { name: /^Failed bg/ }))
    await waitFor(() => expect(screen.getByLabelText("bg output").textContent).toContain("Background output"))
    expect(screen.getByLabelText("bg output").textContent).toContain("Full task")
    expect(inspect).toHaveBeenCalledWith(rows[2])
  })

  it("shows an inspect error", async () => {
    renderRoster({ inspect: vi.fn().mockResolvedValue({ requestId: "r", error: { code: "stale", message: "Artifacts are gone" } }) })
    fireEvent.click(screen.getByRole("button", { name: /^Failed bg/ }))
    await waitFor(() => expect(screen.getByLabelText("bg output").textContent).toBe("Artifacts are gone"))
  })

  it("dismisses a finished row and clears all finished", () => {
    const { onDismiss, onClear } = renderRoster()
    expect(screen.queryByRole("button", { name: "Dismiss scout" })).toBeNull()
    fireEvent.click(screen.getByRole("button", { name: "Dismiss worker" }))
    expect(onDismiss).toHaveBeenCalledWith("b")
    fireEvent.click(screen.getByRole("button", { name: "Clear finished subagents" }))
    expect(onClear).toHaveBeenCalled()
  })
})

const call = (status: string): Record<string, ToolRun> => ({
  call: { running: status === "running", isError: false, result: { content: [], details: { mode: "single", progress: [{ index: 0, agent: "scout", status }] } } },
})

describe("useSubagentRoster", () => {
  it("keeps a finished child until the next run starts, then it stays out", () => {
    const widgets = {}
    const { result, rerender } = renderHook(({ runs, running }) => useSubagentRoster(runs, widgets, running), { initialProps: { runs: call("running"), running: true } })
    expect(result.current.rows.map((row) => row.status)).toEqual(["running"])
    const finished = call("completed")
    rerender({ runs: finished, running: false })
    expect(result.current.rows.map((row) => row.status)).toEqual(["done"])
    rerender({ runs: finished, running: true })
    expect(result.current.rows).toEqual([])
  })

  it("dismisses one child", () => {
    const runs = call("completed")
    const widgets = {}
    const { result } = renderHook(() => useSubagentRoster(runs, widgets, false))
    expect(result.current.rows).toHaveLength(1)
    act(() => result.current.dismiss(result.current.rows[0]!.id))
    expect(result.current.rows).toEqual([])
  })
})
