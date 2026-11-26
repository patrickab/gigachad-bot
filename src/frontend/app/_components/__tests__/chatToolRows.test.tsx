import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import type { ToolCallRecord } from "@/lib/domain/types"
import { ChatToolCall } from "@/features/chat/toolRenderers"
import "../toolRenderers"

// A chat memory call renders as Pi's memory row, which the app shell loads for the chat
describe("chat memory calls", () => {
  const opsCall: ToolCallRecord = {
    id: "mem-1",
    name: "apply_memory_ops",
    arguments: { memory_store: "user", ops: [{ op: "add", path: "preferences/food", text: "likes oat milk" }, { op: "retire", path: "preferences/food", id: "k2p" }] },
    status: "done",
    summary: "2/2 applied",
  }

  it("summarises the ops collapsed and shows one line per op opened", () => {
    render(<ChatToolCall record={opsCall} />)
    expect(screen.getByText("2 changes · preferences/food")).toBeTruthy()
    fireEvent.click(screen.getByRole("button", { expanded: false }))
    const lines = [...document.querySelectorAll("[data-memory-op]")].map((line) => line.textContent)
    expect(lines).toEqual(["+likes oat milk", "−k2pretired"])
    expect(screen.queryByText("ops")).toBeNull()
  })

  it("keeps the first argument for malformed memory arguments", () => {
    render(<ChatToolCall record={{ ...opsCall, name: "read_memories", arguments: { memory_store: "user", paths: "x" } }} />)
    expect(screen.getByText("user")).toBeTruthy()
  })
})
