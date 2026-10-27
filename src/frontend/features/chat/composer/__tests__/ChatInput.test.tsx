import { afterEach, describe, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
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
