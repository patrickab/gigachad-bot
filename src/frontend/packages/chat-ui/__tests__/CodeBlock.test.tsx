import { act, cleanup, fireEvent, render, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { CodeBlock } from "../src/CodeBlock"

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe("CodeBlock", () => {
  it("highlights in the host's color scheme", async () => {
    const view = render(<CodeBlock codeString="const x = 1" language="ts" theme="light" />)
    const block = view.container.querySelector(".gc-code")!
    expect(block.getAttribute("data-theme")).toBe("light")
    expect(block.getAttribute("data-code-language")).toBe("ts")
    expect(block.querySelector(".gc-code-fallback")?.textContent).toBe("const x = 1")
    await waitFor(() => expect(block.querySelector(".gc-code-highlighted pre.shiki")).not.toBeNull())
    view.rerender(<CodeBlock codeString="const x = 1" language="ts" theme="dark" />)
    expect(block.getAttribute("data-theme")).toBe("dark")
  })

  it("leaves text-drawn diagrams unhighlighted", () => {
    const view = render(<CodeBlock codeString={"┌─┐\n└─┘"} language="text" plain theme="dark" />)
    const block = view.container.querySelector(".gc-code")!
    expect(block.classList.contains("gc-code-plain")).toBe(true)
    expect(block.getAttribute("data-code-language")).toBe("")
    expect(view.getByRole("button", { name: "Copy diagram" })).toBeTruthy()
  })

  it("copies the source", async () => {
    const writeText = vi.fn(() => Promise.resolve())
    vi.stubGlobal("navigator", { clipboard: { writeText } })
    const view = render(<CodeBlock codeString="print(1)" language="python" theme="dark" />)
    await act(async () => { fireEvent.click(view.getByRole("button", { name: "Copy code" })) })
    expect(writeText).toHaveBeenCalledWith("print(1)")
    expect(view.container.querySelector(".gc-code-copied")).not.toBeNull()
  })
})
