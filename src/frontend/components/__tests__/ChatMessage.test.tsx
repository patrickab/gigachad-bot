import { fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

vi.mock("@/components/LaTeXMarkdown", () => ({
  LaTeXMarkdown: ({ content }: { content: string }) => <span data-testid="message-content">{content}</span>,
}))

import { ChatMessage } from "@/components/ChatMessage"

afterEach(() => window.getSelection()?.removeAllRanges())

function selectContents(element: Node) {
  const range = document.createRange()
  range.selectNodeContents(element)
  const selection = window.getSelection()!
  selection.removeAllRanges()
  selection.addRange(range)
}

describe("ChatMessage native copy", () => {
  it("copies the original user message when its rendered content is fully selected", () => {
    const content = "First paragraph.\n\nSecond paragraph."
    render(<ChatMessage role="user" content={content} index={0} />)

    const rendered = screen.getByTestId("message-content")
    const message = rendered.parentElement!
    selectContents(message)
    const setData = vi.fn()

    fireEvent.copy(message, { clipboardData: { setData } })

    expect(setData).toHaveBeenCalledWith("text/plain", content)
  })

  it("keeps partial selections under browser control", () => {
    render(<ChatMessage role="user" content="First paragraph." index={0} />)

    const rendered = screen.getByTestId("message-content")
    const text = rendered.firstChild!
    const range = document.createRange()
    range.setStart(text, 0)
    range.setEnd(text, 5)
    const selection = window.getSelection()!
    selection.removeAllRanges()
    selection.addRange(range)
    const setData = vi.fn()

    fireEvent.copy(rendered.parentElement!, { clipboardData: { setData } })

    expect(setData).not.toHaveBeenCalled()
  })
})
