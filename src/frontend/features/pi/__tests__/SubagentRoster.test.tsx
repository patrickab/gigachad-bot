import { afterEach, describe, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { SubagentRoster } from "../SubagentRoster"
import type { SubagentRow } from "../subagents"

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
})
