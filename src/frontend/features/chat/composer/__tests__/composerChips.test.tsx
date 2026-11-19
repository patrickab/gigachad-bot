import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react"
import { ChatInput } from "../ChatInput"
import { stageChip } from "../composerChips"

vi.mock("@/features/chat/EnabledToolsContext", () => ({ useEnabledTools: () => ({ enabledTools: [], toggleTool: () => {} }) }))
vi.mock("@/contexts/SettingsContext", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/contexts/SettingsContext")>(),
  useSettings: () => ({ ocrModel: "" }),
  useNeovimSetting: () => ({ available: false, enabled: false, setPreferred: () => {} }),
}))

const draftOf = () => screen.getByRole("textbox") as HTMLTextAreaElement
const lines = (count: number) => Array.from({ length: count }, (_, index) => `line ${index}`).join("\n")
/** Paste `text` into the draft, true when the composer took it over instead of the textarea. */
const paste = (text: string) => !fireEvent.paste(draftOf(), { clipboardData: { files: [], items: [], getData: () => text } })

describe("composer chips", () => {
  beforeEach(() => {
    const stored = new Map<string, string>()
    vi.stubGlobal("localStorage", {
      getItem: (key: string) => stored.get(key) ?? null,
      setItem: (key: string, value: string) => { stored.set(key, value) },
      removeItem: (key: string) => { stored.delete(key) },
    })
  })
  afterEach(() => { cleanup(); vi.unstubAllGlobals() })

  it("turns a large paste into a chip and leaves a small one to the draft", () => {
    render(<ChatInput chatId="c1" onSend={() => {}} />)

    expect(paste("short")).toBe(false)
    expect(paste(`${lines(40)}\n`)).toBe(true)

    expect(screen.getByText("pasted text · 40 lines")).toBeInTheDocument()
    expect(draftOf().value).toBe("")
  })

  it("sends without typed text and appends chip text verbatim in staging order", () => {
    const onSend = vi.fn()
    render(<ChatInput chatId="c1" draftKey="chips:send" onSend={onSend} />)
    const big = lines(31)
    paste(big)
    act(() => { stageChip("chips:send", { label: "diff hunk", text: "@@ -1 +1 @@\n-a\n+b" }) })

    fireEvent.keyDown(draftOf(), { key: "Enter" })
    expect(onSend).toHaveBeenLastCalledWith(`${big}\n\n@@ -1 +1 @@\n-a\n+b`, [])
    expect(screen.queryByText("diff hunk")).toBeNull()

    act(() => { stageChip("chips:send", { label: "test output", text: "1 failed" }) })
    fireEvent.change(draftOf(), { target: { value: "  why?  " } })
    fireEvent.keyDown(draftOf(), { key: "Enter" })
    expect(onSend).toHaveBeenLastCalledWith("why?\n\n1 failed", [])
  })

  it("removes a chip from the message", () => {
    const onSend = vi.fn()
    render(<ChatInput chatId="c1" draftKey="chips:remove" onSend={onSend} />)
    act(() => {
      stageChip("chips:remove", { label: "first", text: "one" })
      stageChip("chips:remove", { label: "second", text: "two" })
    })

    fireEvent.click(screen.getByRole("button", { name: "Remove first" }))
    expect(screen.queryByText("first")).toBeNull()
    fireEvent.keyDown(draftOf(), { key: "Enter" })
    expect(onSend).toHaveBeenCalledWith("two", [])
  })

  it("keeps chips per conversation across switching and closing", () => {
    const first = render(<ChatInput chatId="c1" draftKey="chips:a" onSend={() => {}} />)
    act(() => { stageChip("chips:a", { label: "kept", text: "context" }) })
    first.rerender(<ChatInput chatId="c1" draftKey="chips:b" onSend={() => {}} />)
    expect(screen.queryByText("kept")).toBeNull()

    first.rerender(<ChatInput chatId="c1" draftKey="chips:a" onSend={() => {}} />)
    expect(screen.getByText("kept")).toBeInTheDocument()
    cleanup()

    render(<ChatInput chatId="c2" draftKey="chips:a" onSend={() => {}} />)
    expect(screen.getByText("kept")).toBeInTheDocument()
  })
})
