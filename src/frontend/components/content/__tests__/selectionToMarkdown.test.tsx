import { fireEvent, render } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"
import { selectionToMarkdown } from "@gigachad/chat-ui"

/** Render `content`, select `target` (or everything) and return the markdown a copy would produce. */
function copiedMarkdown(content: string, selector?: string): string | null {
  const { container } = render(<LaTeXMarkdown content={content} />)
  const root = container.firstElementChild as HTMLElement
  const range = document.createRange()
  if (selector) range.selectNodeContents(root.querySelector(selector)!)
  else range.selectNodeContents(root)
  const selection = window.getSelection()!
  selection.removeAllRanges()
  selection.addRange(range)
  return selectionToMarkdown(selection, root)
}

describe("selectionToMarkdown", () => {
  it("restores the heading level of a selected heading", () => {
    expect(copiedMarkdown("## Key intuition\n\nbody", "h2")).toBe("## Key intuition")
  })

  it("restores bold text selected on its own", () => {
    expect(copiedMarkdown("a **bold** word", "strong")).toBe("**bold**")
  })

  it("returns the full LaTeX source of inline and display math", () => {
    expect(copiedMarkdown(String.raw`Rule $\theta_{t+1}$ here`, ".katex")).toBe(String.raw`$\theta_{t+1}$`)
    expect(copiedMarkdown("$$\nE = mc^2\n$$", ".katex-display")).toBe("$$\nE = mc^2\n$$")
  })

  it("keeps list structure and code fences across a multi-block selection", () => {
    const markdown = copiedMarkdown("- one\n- two\n\n```python\nprint(1)\n```")
    expect(markdown).toBe("- one\n- two\n\n```python\nprint(1)\n```")
  })

  it("keeps the number of a selected ordered list item", () => {
    expect(copiedMarkdown("1. a\n2. b\n3. c", "li:nth-child(2)")).toBe("2. b")
  })

  it("returns plain text unchanged for a plain selection", () => {
    expect(copiedMarkdown("snake_case and 2 * 3", "p")).toBe("snake_case and 2 * 3")
  })
})

describe("copying from a rendered message", () => {
  it("copies the exact source when the whole message is selected", () => {
    const content = "First _paragraph_.\n\n* item"
    const { container } = render(<LaTeXMarkdown content={content} />)
    const root = container.firstElementChild as HTMLElement
    const range = document.createRange()
    range.selectNodeContents(root)
    window.getSelection()!.removeAllRanges()
    window.getSelection()!.addRange(range)
    const setData = vi.fn()

    fireEvent.copy(root, { clipboardData: { setData } })

    expect(setData).toHaveBeenCalledWith("text/plain", content)
  })
})
