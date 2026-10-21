import { cleanup, fireEvent, render } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { TurnAction, TurnRegion } from "../src/Turn"

afterEach(() => {
  cleanup()
  window.getSelection()?.removeAllRanges()
})

describe("expanded question collapse", () => {
  it.each(["text", "avatar", "padding", "chevron"])("collapses once when clicking %s", (target) => {
    const onCollapse = vi.fn()
    const view = render(<TurnRegion role="question" onCollapse={onCollapse}><p>Question text</p></TurnRegion>)
    const region = view.container.querySelector(".gc-turn-region")!
    const targets = {
      text: view.getByText("Question text"),
      avatar: region.querySelector(".gc-turn-avatar svg")!,
      padding: region,
      chevron: view.getByRole("button", { name: "Collapse question and answer" }),
    }
    fireEvent.click(targets[target as keyof typeof targets])
    expect(onCollapse).toHaveBeenCalledTimes(1)
  })

  it("preserves nested links, attachments and actions", () => {
    const onCollapse = vi.fn()
    const onAttachment = vi.fn()
    const onAction = vi.fn()
    const view = render(
      <TurnRegion role="question" onCollapse={onCollapse} actions={<TurnAction label="Copy question" icon={<span>Copy</span>} onClick={onAction} />}>
        <a href="#file"><span>File link</span></a>
        <button onClick={onAttachment}><span>Attachment</span></button>
      </TurnRegion>,
    )
    fireEvent.click(view.getByText("File link"))
    fireEvent.click(view.getByText("Attachment"))
    fireEvent.click(view.getByRole("button", { name: "Copy question" }))
    expect(onAttachment).toHaveBeenCalledTimes(1)
    expect(onAction).toHaveBeenCalledTimes(1)
    expect(onCollapse).not.toHaveBeenCalled()
  })

  it("keeps selected question text expanded", () => {
    const onCollapse = vi.fn()
    const view = render(<TurnRegion role="question" onCollapse={onCollapse}><p>Question text</p></TurnRegion>)
    const text = view.getByText("Question text")
    const range = document.createRange()
    range.selectNodeContents(text)
    window.getSelection()!.addRange(range)
    fireEvent.click(text)
    expect(onCollapse).not.toHaveBeenCalled()
  })
})
