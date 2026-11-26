import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { TurnAnswer, TurnQuestion } from "../TurnQuestion"

function Question(props: Partial<Parameters<typeof TurnQuestion>[0]>) {
  return <TurnQuestion text="Prompt" {...props}><p>Prompt</p></TurnQuestion>
}

const actionNames = (group: string) => {
  const region = screen.queryByRole("group", { name: group })
  return region ? within(region).getAllByRole("button").map((button) => button.getAttribute("aria-label")) : []
}

/** The dialog focuses its confirm button after its enter animation. */
const settleDialog = () => act(() => { vi.advanceTimersByTime(100) })

beforeEach(() => { vi.useFakeTimers() })
afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

describe("TurnQuestion actions", () => {
  it("offers copy, edit and delete on a settled question", () => {
    render(<Question onEdit={() => {}} onDelete={() => {}} />)
    expect(actionNames("Question actions")).toEqual(["Copy message", "Edit message", "Delete turn"])
  })

  it("offers none on a live turn", () => {
    render(<Question live onEdit={() => {}} onDelete={() => {}} />)
    expect(actionNames("Question actions")).toEqual([])
  })

  it("offers copy only without handlers", () => {
    render(<Question />)
    expect(actionNames("Question actions")).toEqual(["Copy message"])
  })
})

describe("TurnAnswer actions", () => {
  it("offers copy and branch when settled, none when live, copy only without a branch", () => {
    const { rerender } = render(<TurnAnswer text="Answer" branchLabel="Branch" onBranch={() => {}}><p>Answer</p></TurnAnswer>)
    expect(actionNames("Answer actions")).toEqual(["Copy message", "Branch"])
    rerender(<TurnAnswer text="Answer" branchLabel="Branch" onBranch={() => {}} live><p>Answer</p></TurnAnswer>)
    expect(actionNames("Answer actions")).toEqual([])
    rerender(<TurnAnswer text="Answer" branchLabel="Branch"><p>Answer</p></TurnAnswer>)
    expect(actionNames("Answer actions")).toEqual(["Copy message"])
  })
})

describe("TurnQuestion editor", () => {
  const openEditor = () => {
    fireEvent.click(screen.getByRole("button", { name: "Edit message" }))
    return screen.getByRole("textbox", { name: "Edit question" }) as HTMLTextAreaElement
  }

  it("holds the prompt, focused, and confirms the new text with Enter", () => {
    const onEdit = vi.fn()
    render(<Question onEdit={onEdit} />)
    const editor = openEditor()
    expect(editor.value).toBe("Prompt")
    expect(document.activeElement).toBe(editor)
    fireEvent.change(editor, { target: { value: "New prompt" } })
    fireEvent.keyDown(editor, { key: "Enter" })
    expect(onEdit).toHaveBeenCalledWith("New prompt")
    expect(screen.queryByRole("textbox")).toBeNull()
  })

  it("keeps Shift+Enter for a newline", () => {
    const onEdit = vi.fn()
    render(<Question onEdit={onEdit} />)
    const editor = openEditor()
    fireEvent.keyDown(editor, { key: "Enter", shiftKey: true })
    expect(onEdit).not.toHaveBeenCalled()
    expect(screen.getByRole("textbox")).toBeTruthy()
  })

  it("cancels with Escape", () => {
    const onEdit = vi.fn()
    render(<Question onEdit={onEdit} />)
    const editor = openEditor()
    fireEvent.change(editor, { target: { value: "New prompt" } })
    fireEvent.keyDown(editor, { key: "Escape" })
    expect(onEdit).not.toHaveBeenCalled()
    expect(screen.queryByRole("textbox")).toBeNull()
    expect(screen.getByText("Prompt")).toBeTruthy()
  })

  it.each([["unchanged", "Prompt"], ["empty", "  "]])("cancels an %s text", (_case, value) => {
    const onEdit = vi.fn()
    render(<Question onEdit={onEdit} laterTurns={2} />)
    const editor = openEditor()
    fireEvent.change(editor, { target: { value } })
    fireEvent.keyDown(editor, { key: "Enter" })
    expect(onEdit).not.toHaveBeenCalled()
    expect(screen.queryByText(/later turns/)).toBeNull()
    expect(screen.queryByRole("textbox")).toBeNull()
  })

  it("asks first with later turns, naming how many go, and confirms with Enter", () => {
    const onEdit = vi.fn()
    render(<Question onEdit={onEdit} laterTurns={2} />)
    const editor = openEditor()
    fireEvent.change(editor, { target: { value: "New prompt" } })
    fireEvent.keyDown(editor, { key: "Enter" })
    expect(onEdit).not.toHaveBeenCalled()
    expect(screen.getByText("The edit also removes the 2 later turns.")).toBeTruthy()
    settleDialog()
    fireEvent.keyDown(screen.getByRole("button", { name: "Edit" }), { key: "Enter" })
    expect(onEdit).toHaveBeenCalledWith("New prompt")
    expect(screen.queryByRole("textbox")).toBeNull()
  })

  it("goes back to the editor when the confirmation is cancelled with Escape", () => {
    const onEdit = vi.fn()
    render(<Question onEdit={onEdit} laterTurns={1} />)
    const editor = openEditor()
    fireEvent.change(editor, { target: { value: "New prompt" } })
    fireEvent.keyDown(editor, { key: "Enter" })
    expect(screen.getByText("The edit also removes the 1 later turn.")).toBeTruthy()
    fireEvent.keyDown(window, { key: "Escape" })
    settleDialog()
    expect(onEdit).not.toHaveBeenCalled()
    expect((screen.getByRole("textbox") as HTMLTextAreaElement).value).toBe("New prompt")
  })
})

describe("TurnQuestion delete", () => {
  it("asks first and deletes on confirm", () => {
    const onDelete = vi.fn()
    render(<Question onDelete={onDelete} />)
    fireEvent.click(screen.getByRole("button", { name: "Delete turn" }))
    expect(onDelete).not.toHaveBeenCalled()
    settleDialog()
    fireEvent.keyDown(screen.getByRole("button", { name: "Delete" }), { key: "Enter" })
    expect(onDelete).toHaveBeenCalledTimes(1)
  })

  it("deletes nothing when cancelled", () => {
    const onDelete = vi.fn()
    render(<Question onDelete={onDelete} />)
    fireEvent.click(screen.getByRole("button", { name: "Delete turn" }))
    settleDialog()
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }))
    fireEvent.keyDown(window, { key: "Escape" })
    expect(onDelete).not.toHaveBeenCalled()
  })
})
