import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeAll, expect, it, vi } from "vitest"
import { OmpSelect } from "../OmpSelect"

const request = {
  id: "r1",
  title: "Files changed by this session",
  options: ["./scripts/setup_agent_sandbox.sh", "./README.md", "[agent session] ~/.omp/agent/sessions/a/2026.jsonl"],
}

function renderSelect() {
  const onAnswer = vi.fn()
  render(<OmpSelect request={request} onAnswer={onAnswer} />)
  return { onAnswer, input: screen.getByRole("combobox", { name: request.title }) }
}

// jsdom does not lay out, so it has no scrollIntoView
beforeAll(() => { Element.prototype.scrollIntoView = vi.fn() })
afterEach(cleanup)

it("ranks a file name match first and Enter picks it", () => {
  const { onAnswer, input } = renderSelect()
  fireEvent.change(input, { target: { value: "rdme" } })
  expect(screen.getAllByRole("option").map((option) => option.textContent)).toEqual(["README.md."])
  fireEvent.keyDown(input, { key: "Enter" })
  expect(onAnswer).toHaveBeenCalledWith("./README.md")
})

it("arrows move the selection within the list and Enter picks the selected option", () => {
  const { onAnswer, input } = renderSelect()
  fireEvent.keyDown(input, { key: "ArrowUp" })
  fireEvent.keyDown(input, { key: "ArrowDown" })
  fireEvent.keyDown(input, { key: "ArrowDown" })
  fireEvent.keyDown(input, { key: "ArrowDown" })
  expect(screen.getByRole("option", { selected: true })).toHaveTextContent("2026.jsonl")
  fireEvent.keyDown(input, { key: "Enter" })
  expect(onAnswer).toHaveBeenCalledWith(request.options[2])
})

it("Escape cancels, Enter without a match answers nothing", () => {
  const { onAnswer, input } = renderSelect()
  fireEvent.change(input, { target: { value: "zzz" } })
  fireEvent.keyDown(input, { key: "Enter" })
  expect(onAnswer).not.toHaveBeenCalled()
  fireEvent.keyDown(input, { key: "Escape" })
  expect(onAnswer).toHaveBeenCalledWith(null)
})
