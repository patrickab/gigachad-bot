import { afterEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react"
import { useRef, useState } from "react"
import { File as FileIcon } from "lucide-react"
import { CompletionMenu, useCompletions, type CompleteDraft, type ComposerCompletion } from "../CompletionMenu"

const fileRow = (path: string, from: number, to: number): ComposerCompletion => ({
  label: path, icon: FileIcon, from, to, insert: `@${path} `, caretOffset: path.length + 2,
})

/** Minimal composer: a textarea wired like ChatInput wires it. */
function Composer({ complete, onSubmit = () => {} }: { complete: CompleteDraft; onSubmit?: (text: string) => void }) {
  const [text, setText] = useState("")
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const completions = useCompletions(complete, textareaRef, setText)
  return (
    <>
      <CompletionMenu items={completions.items} activeIndex={completions.activeIndex} onHover={completions.setActiveIndex} onApply={(item) => completions.apply(text, item)} />
      <textarea
        aria-label="draft"
        ref={textareaRef}
        value={text}
        onChange={(event) => { setText(event.target.value); completions.request(event.target.value, event.target.selectionStart) }}
        onKeyDown={(event) => { if (!completions.handleKey(event, text) && event.key === "Enter") onSubmit(text) }}
      />
    </>
  )
}

async function type(value: string) {
  const draft = screen.getByRole("textbox", { name: "draft" }) as HTMLTextAreaElement
  fireEvent.change(draft, { target: { value, selectionStart: value.length } })
  await act(() => vi.advanceTimersByTimeAsync(100))
  return draft
}

describe("useCompletions", () => {
  afterEach(() => { cleanup(); vi.useRealTimers() })

  it("replaces the token with the chosen row and keeps the text after it", async () => {
    vi.useFakeTimers()
    const complete = vi.fn(async (text: string) => (text.endsWith("@ma") ? [fileRow("lib/a.ts", 4, 7), fileRow("main.ts", 4, 7)] : []))
    render(<Composer complete={complete} />)

    const draft = await type("see @ma")
    fireEvent.keyDown(draft, { key: "ArrowDown" })
    fireEvent.keyDown(draft, { key: "Enter" })

    expect(draft.value).toBe("see @main.ts ")
    await act(() => vi.advanceTimersByTimeAsync(100))
    expect(screen.queryByRole("listbox")).toBeNull()
  })

  it("takes Enter for the menu instead of submitting, Escape closes it", async () => {
    vi.useFakeTimers()
    const submit = vi.fn()
    render(<Composer complete={async () => [fileRow("a.ts", 0, 2)]} onSubmit={submit} />)

    const draft = await type("@a")
    fireEvent.keyDown(draft, { key: "Escape" })
    expect(screen.queryByRole("listbox")).toBeNull()
    fireEvent.keyDown(draft, { key: "Enter" })

    expect(submit).toHaveBeenCalledWith("@a")
  })

  it("shows only the answer to the latest draft", async () => {
    vi.useFakeTimers()
    let resolveFirst: (items: ComposerCompletion[]) => void = () => {}
    const complete = vi.fn((text: string) => (text === "@a"
      ? new Promise<ComposerCompletion[]>((resolve) => { resolveFirst = resolve })
      : Promise.resolve([fileRow("ab.ts", 0, 3)])))
    render(<Composer complete={complete} />)

    await type("@a")
    await type("@ab")
    await act(async () => resolveFirst([fileRow("stale.ts", 0, 2)]))

    expect(screen.getAllByRole("option").map((option) => option.textContent)).toEqual(["ab.ts"])
  })
})
