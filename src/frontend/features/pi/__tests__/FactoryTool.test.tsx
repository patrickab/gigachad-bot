import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

vi.mock("@/components/content/LaTeXMarkdown", () => ({
  LaTeXMarkdown: ({ content }: { content: string }) => <div data-testid="markdown">{content}</div>,
}))

import "../FactoryTool"
import "../MemoryTool"
import { renderToolCall } from "../renderers"
import type { ToolCall, ToolRun } from "../types"

const done = (text: string, isError = false): ToolRun => ({ result: { content: [{ type: "text", text }] }, isError, running: false })
const show = (name: string, args: Record<string, unknown>, run?: ToolRun) =>
  render(<>{renderToolCall({ type: "toolCall", id: "c1", name, arguments: args } as ToolCall, run, () => <p>default row</p>)}</>)
const open = (summary: RegExp) => fireEvent.click(screen.getByRole("button", { name: summary }))

const wave = {
  action: "waves.start",
  repo: ".",
  tag: "board/w4",
  title: "feat(factory): a board with room",
  request: "Check the columns line up",
  tasks: [
    { key: "wave", model: "small", title: "feat(factory): a wave's header", request: "…" },
    { key: "layout", title: "feat(factory): three columns", request: "…", after: ["wave"] },
  ],
}

describe("factory row", () => {
  it("shows the verb, what the call names and the result's first line collapsed", () => {
    show("factory", wave, done("started · building 0/2\nwave 3f2a91c0"))
    expect(screen.getByText("start wave · board/w4 · 2 tasks")).toBeTruthy()
    expect(screen.getByText("started · building 0/2")).toBeTruthy()
    expect(screen.queryByText("feat(factory): a board with room")).toBeNull()
  })

  it("opens to the title, one line per task, long text and the result behind disclosures", () => {
    show("factory", wave, done("started · building 0/2\nwave 3f2a91c0"))
    open(/start wave/)
    expect(screen.getByText("feat(factory): a board with room")).toBeTruthy()
    const layout = document.querySelector("[data-wave-task='layout']")!
    expect(layout.textContent).toBe("layoutbigfeat(factory): three columnsafter wave")
    expect(document.querySelector("[data-wave-task='wave']")!.textContent).toContain("small")
    expect(screen.queryByText("Check the columns line up")).toBeNull()
    fireEvent.click(screen.getByRole("button", { name: "Reviewer instructions" }))
    expect(screen.getByText("Check the columns line up")).toBeTruthy()
    fireEvent.click(screen.getByRole("button", { name: "Result" }))
    expect(screen.getByText(/wave 3f2a91c0/)).toBeTruthy()
    expect(screen.queryByText(/"action"/)).toBeNull()
  })

  it("names a task call by its title and shows an error result's first line", () => {
    show("factory", { action: "tasks.start", repo: ".", title: "fix: a bug", request: "Fix it" }, done("refused: no factory models are set", true))
    expect(screen.getByText("start task · fix: a bug")).toBeTruthy()
    expect(screen.getByText("refused: no factory models are set").className).toContain("text-danger")
    expect(screen.getByText("failed")).toBeTruthy()
  })

  it("uses the action name for an unknown action and a short id for a known one", () => {
    show("factory", { action: "waves.add", id: "3f2a91c0deadbeef", tasks: [{ key: "fix", title: "fix" }] }, done("building 5/6"))
    expect(screen.getByText("add tasks · 3f2a91c0 · 1 task")).toBeTruthy()
    show("factory", { action: "things.do" })
    expect(screen.getByText("things.do")).toBeTruthy()
  })

  it("falls back to the default row for a missing action or unreadable tasks", () => {
    show("factory", { repo: "." })
    show("factory", { action: "waves.start", tasks: "two" })
    expect(screen.getAllByText("default row")).toHaveLength(2)
  })
})

describe("memory rows", () => {
  it("summarises a read and renders its result as markdown", () => {
    show("read_memories", { memory_store: "user", paths: ["preferences/food", "people/anna"] }, done("# food\n- oat milk"))
    expect(screen.getByText("read · user · preferences/food, people/anna")).toBeTruthy()
    open(/read · user/)
    fireEvent.click(screen.getByRole("button", { name: "Result" }))
    expect(screen.getByTestId("markdown").textContent).toBe("# food\n- oat milk")
  })

  it("matches the MCP name and lists as plain text", () => {
    show("mcp__memory__list_memories", { memory_store: "gigachad-bot" }, done("plans/ 2 memories"))
    expect(screen.getByText("list · gigachad-bot")).toBeTruthy()
    open(/list · gigachad-bot/)
    fireEvent.click(screen.getByRole("button", { name: "Result" }))
    expect(screen.getByText("plans/ 2 memories")).toBeTruthy()
    expect(screen.queryByTestId("markdown")).toBeNull()
  })

  it("shows one line per op when opened", () => {
    show("apply_memory_ops", {
      memory_store: "user",
      ops: [
        { op: "add", path: "preferences/food", text: "likes oat milk in coffee" },
        { op: "update", path: "preferences/food", id: "a3f", text: "allergic to hazelnuts" },
        { op: "retire", path: "preferences/food", id: "k2p" },
      ],
    }, done("ok"))
    expect(screen.getByText("3 changes · preferences/food")).toBeTruthy()
    expect(document.querySelector("[data-memory-op]")).toBeNull()
    open(/3 changes/)
    const lines = [...document.querySelectorAll("[data-memory-op]")].map((line) => line.textContent)
    expect(lines).toEqual(["+likes oat milk in coffee", "~a3fallergic to hazelnuts", "−k2pretired"])
  })

  it("shows an error result in the danger colour and falls back on malformed arguments", () => {
    show("read_memories", { memory_store: "user", paths: ["x"] }, done("error: not_found: x", true))
    open(/read · user · x/)
    fireEvent.click(screen.getByRole("button", { name: "Result" }))
    expect(screen.getByText("error: not_found: x").closest("[data-error]")).toBeTruthy()
    show("read_memories", { paths: [] })
    show("apply_memory_ops", { ops: [{ path: "x" }] })
    expect(screen.getAllByText("default row")).toHaveLength(2)
  })
})
