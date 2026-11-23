import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import type { ReactNode } from "react"
import { factoryTask } from "./factoryTask"

const api = vi.hoisted(() => ({ callFactory: vi.fn<(action: string, params: Record<string, unknown>) => Promise<unknown>>() }))

vi.mock("@/lib/backend/api", () => api)
vi.mock("@/features/editor/FilePanel", () => ({
  FilePanel: ({ label, children }: { label: string; children: ReactNode }) => <section aria-label={label}>{children}</section>,
}))

import { AgentPanel } from "@/features/factory/AgentPanel"

function renderPanel(canMessage = true) {
  const onSent = vi.fn()
  const entries = [{ at: "", kind: "coder", data: { text: "Added the toggle." } }] as const
  render(<AgentPanel task={factoryTask("t1")} entries={entries} timelineError="" canMessage={canMessage} onSent={onSent} onOpenTask={vi.fn()} onClose={vi.fn()} />)
  return { onSent, box: screen.getByLabelText("Message to the coder") }
}

beforeEach(() => {
  api.callFactory.mockReset().mockResolvedValue({})
  // The composer fits its height to the text
  vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} })
})

afterEach(() => {
  vi.unstubAllGlobals()
})

describe("AgentPanel", () => {
  it("shows the task's timeline above the message box", () => {
    renderPanel()
    expect(screen.getByText("Added the toggle.")).toBeTruthy()
  })

  it("sends a message to the core on Enter, a newline with Shift+Enter, and clears the box once sent", async () => {
    const { onSent, box } = renderPanel()
    fireEvent.change(box, { target: { value: "Also the README" } })
    fireEvent.keyDown(box, { key: "Enter", shiftKey: true })
    expect(api.callFactory).not.toHaveBeenCalled()

    fireEvent.keyDown(box, { key: "Enter" })
    await waitFor(() => expect(api.callFactory).toHaveBeenCalledWith("tasks.message", { id: "t1", text: "Also the README" }))
    await waitFor(() => expect(box).toHaveProperty("value", ""))
    expect(onSent).toHaveBeenCalledTimes(1)
  })

  it("keeps the text and shows the core's refusal", async () => {
    api.callFactory.mockRejectedValue(new Error("task t1 is merging, send the message when the merge has ended"))
    const { onSent, box } = renderPanel()
    fireEvent.change(box, { target: { value: "Wait" } })
    fireEvent.keyDown(box, { key: "Enter" })

    expect(await screen.findByRole("alert")).toHaveTextContent("task t1 is merging")
    expect(box).toHaveProperty("value", "Wait")
    expect(onSent).not.toHaveBeenCalled()
  })

  it("disables the box while the task takes no message", () => {
    const { box } = renderPanel(false)
    expect(box).toBeDisabled()
  })
})
