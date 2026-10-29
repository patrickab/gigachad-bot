import { afterEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react"
import { File as FileIcon } from "lucide-react"
import type { Attachment } from "@/lib/domain/types"
import { TabContext } from "@/contexts/TabContext"
import { nextTab } from "@/features/navigation/tabs"
import { ChatInput, type ChatContext } from "../ChatInput"

vi.mock("@/features/chat/EnabledToolsContext", () => ({ useEnabledTools: () => ({ enabledTools: [], toggleTool: () => {} }) }))
vi.mock("@/contexts/SettingsContext", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/contexts/SettingsContext")>(),
  useSettings: () => ({ ocrModel: "" }),
  useNeovimSetting: () => ({ available: false, enabled: false, setPreferred: () => {} }),
}))
vi.mock("@/contexts/ProjectContext", () => ({
  useDisplayName: () => (slug: string | null, path: string) => slug === "proj" && path === "project/proj/document/plan.md" ? "Roadmap.md" : undefined,
}))

const notes = (content: string): Attachment => ({ name: "notes.md", mime: "text/markdown", url: "", content, active: true })

function renderComposer(context: ChatContext) {
  return render(<ChatInput chatId="c1" onSend={() => {}} context={context} />)
}

function contextOf(attachments: ChatContext["attachments"]): ChatContext {
  return { attachments, onToggleActive: vi.fn(), onRemove: vi.fn(), onOpen: vi.fn() }
}

describe("ChatInput context chips", () => {
  afterEach(cleanup)

  it("routes open, toggle and remove to the message that carries the attachment", () => {
    // The same file name attached to two messages: each chip acts on its own message.
    const second = notes("v2")
    const context = contextOf([{ messageIndex: 0, attachment: notes("v1") }, { messageIndex: 2, attachment: second }])
    renderComposer(context)

    fireEvent.click(screen.getAllByRole("button", { name: "Preview notes.md" })[1])
    fireEvent.click(screen.getAllByRole("button", { name: "Exclude notes.md from context" })[1])
    fireEvent.click(screen.getAllByRole("button", { name: "Remove notes.md" })[1])

    expect(context.onOpen).toHaveBeenCalledWith(2, second)
    expect(context.onToggleActive).toHaveBeenCalledWith(2, "notes.md")
    expect(context.onRemove).toHaveBeenCalledWith(2, "notes.md")
  })

  it("labels a referenced project file by the display name its chat's project gives it", () => {
    const plan: Attachment = { name: "plan.md", mime: "text/markdown", url: "", content: "", active: true, vaultPath: "project/proj/document/plan.md" }
    const context = contextOf([{ messageIndex: 0, attachment: plan }])
    render(
      <TabContext.Provider value={{ tab: { ...nextTab(), projectSlug: "proj" }, isActive: true, setModeLabel: vi.fn(), updateConfig: vi.fn(), setAppMode: vi.fn(), setWorkspaceItem: vi.fn(), setHistoryFile: vi.fn(), setTitle: vi.fn() }}>
        <ChatInput chatId="c1" onSend={() => {}} context={context} />
      </TabContext.Provider>,
    )

    expect(screen.getByRole("button", { name: "Preview Roadmap.md" })).toHaveTextContent("Roadmap.md")
    fireEvent.click(screen.getByRole("button", { name: "Remove Roadmap.md" }))
    expect(context.onRemove).toHaveBeenCalledWith(0, "plan.md")
  })
})

describe("ChatInput Escape", () => {
  afterEach(() => { cleanup(); vi.useRealTimers() })

  it("closes an open completion menu first, then leaves the draft", async () => {
    vi.useFakeTimers()
    const complete = async () => [{ label: "a.ts", icon: FileIcon, from: 0, to: 2, insert: "@a.ts ", caretOffset: 6 }]
    render(<ChatInput chatId="c1" onSend={() => {}} complete={complete} />)
    const draft = screen.getByRole("textbox") as HTMLTextAreaElement
    draft.focus()
    fireEvent.change(draft, { target: { value: "@a", selectionStart: 2 } })
    await act(() => vi.advanceTimersByTimeAsync(100))
    expect(screen.getByRole("listbox")).toBeInTheDocument()

    fireEvent.keyDown(draft, { key: "Escape" })
    expect(screen.queryByRole("listbox")).toBeNull()
    expect(draft).toHaveFocus()

    fireEvent.keyDown(draft, { key: "Escape" })
    expect(draft).not.toHaveFocus()
    expect(draft.value).toBe("@a")
  })
})

describe("ChatInput picker", () => {
  afterEach(cleanup)

  const options = ["./scripts/setup.sh", "./README.md", "./notes/readme-old.md"]

  function renderPicker() {
    const onSend = vi.fn()
    const picker = {
      label: "Files changed by this session",
      items: (query: string) => options.filter((option) => option.toLowerCase().includes(query)).map((option) => ({ key: option, label: option })),
      onPick: vi.fn(),
      onCancel: vi.fn(),
    }
    render(<ChatInput chatId="c1" onSend={onSend} picker={picker} />)
    return { onSend, picker, draft: screen.getByRole("textbox") as HTMLTextAreaElement }
  }

  it("takes focus, filters by the draft and picks the arrowed row on Enter without sending it", () => {
    const { onSend, picker, draft } = renderPicker()
    expect(draft).toHaveFocus()
    fireEvent.change(draft, { target: { value: "readme" } })
    expect(screen.getAllByRole("option").map((option) => option.textContent)).toEqual(["./README.md", "./notes/readme-old.md"])
    fireEvent.keyDown(draft, { key: "ArrowDown" })
    fireEvent.keyDown(draft, { key: "Enter" })
    expect(picker.onPick).toHaveBeenCalledWith("./notes/readme-old.md")
    expect(onSend).not.toHaveBeenCalled()
    expect(draft.value).toBe("")
  })

  it("keeps the menu open without matches, Enter then neither picks nor sends, Escape cancels", () => {
    const { onSend, picker, draft } = renderPicker()
    fireEvent.change(draft, { target: { value: "zzz" } })
    expect(screen.getByRole("listbox", { name: "Files changed by this session" })).toHaveTextContent("No matches")
    fireEvent.keyDown(draft, { key: "Enter" })
    expect(picker.onPick).not.toHaveBeenCalled()
    expect(onSend).not.toHaveBeenCalled()
    fireEvent.keyDown(draft, { key: "Escape" })
    expect(picker.onCancel).toHaveBeenCalledOnce()
    expect(draft.value).toBe("")
  })
})
