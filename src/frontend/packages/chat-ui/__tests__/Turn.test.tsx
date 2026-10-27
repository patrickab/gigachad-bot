import { cleanup, fireEvent, render } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { TurnAction, TurnRegion } from "../src/Turn"

afterEach(() => {
  cleanup()
  window.getSelection()?.removeAllRanges()
})

describe("expanded question collapse", () => {
  it.each(["avatar", "strip", "chevron"])("collapses once when clicking %s", (target) => {
    const onCollapse = vi.fn()
    const view = render(<TurnRegion role="question" onCollapse={onCollapse}><p>Question text</p></TurnRegion>)
    const region = view.container.querySelector(".gc-turn-region")!
    const targets = {
      avatar: region.querySelector(".gc-turn-avatar svg")!,
      strip: region.querySelector(".gc-turn-collapse-strip")!,
      chevron: region.querySelector(".gc-turn-collapse")!,
    }
    fireEvent.click(targets[target as keyof typeof targets])
    expect(onCollapse).toHaveBeenCalledTimes(1)
  })

  it.each(["text", "padding"])("keeps the card expanded when clicking %s", (target) => {
    const onCollapse = vi.fn()
    const view = render(<TurnRegion role="question" onCollapse={onCollapse}><p>Question text</p></TurnRegion>)
    fireEvent.click(target === "text" ? view.getByText("Question text") : view.container.querySelector(".gc-turn-region")!)
    expect(onCollapse).not.toHaveBeenCalled()
  })

  it("keeps nested links, attachments and actions independent", () => {
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
})
