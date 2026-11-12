import { afterEach, describe, expect, it, vi } from "vitest"
import { useState, type ReactNode } from "react"
import { act, cleanup, render, screen } from "@testing-library/react"
import { TabContext, type Tab } from "@/contexts/TabContext"
import type { Message } from "@/lib/domain/types"
import { TabContent } from "../TabContent"

const probe = vi.hoisted(() => ({ sidebarRenders: 0, setMessages: null as ((messages: Message[]) => void) | null }))

vi.mock("@/features/navigation/Sidebar", () => ({
  Sidebar: function Sidebar() { probe.sidebarRenders++; return <nav /> },
}))
vi.mock("@/features/chat/ChatContainer", () => ({
  ChatContainer: ({ messages }: { messages: Message[] }) => <div data-testid="chat">{messages.map((message) => message.content).join("")}</div>,
}))
vi.mock("../WorkspaceSurface", () => ({ WorkspaceSurface: ({ chatBody }: { chatBody: ReactNode }) => <>{chatBody}</> }))
vi.mock("../MemoryCommandLayer", () => ({ MemoryCommandLayer: () => null }))
vi.mock("@/features/navigation/TabPicker", () => ({ TabPicker: () => null }))
vi.mock("@/features/editor/FilePanel", () => ({ FilePanelSlot: () => null }))
vi.mock("@/features/settings/PromptEditor", () => ({ PromptEditor: () => null }))
vi.mock("@/components/ui/SaveAsModal", () => ({ SaveAsModal: () => null }))
vi.mock("@/features/omp/WorkspacePicker", () => ({ WorkspacePicker: () => null }))
vi.mock("@/features/navigation/SidebarContext", () => ({ useSidebar: () => ({ collapsed: false, toggleCollapsed: () => {} }) }))
vi.mock("@/features/chat/EnabledToolsContext", () => ({ useEnabledTools: () => ({ researchEnabled: false }) }))
vi.mock("@/hooks/useResource", () => ({ useResource: () => undefined }))
// The tab's chat, its messages replaced by the test as a stream would append tokens.
vi.mock("../useTabChat", () => ({
  useTabChat: () => {
    const [messages, setMessages] = useState<Message[]>([])
    probe.setMessages = setMessages
    return {
      chat: { messages, isStreaming: true, models: null, totalUsage: undefined },
      filePicker: { standaloneFile: null, rootFiles: { enabled: false } },
      docs: {}, persistence: {}, turns: {}, memory: {}, menuItems: [],
    }
  },
}))

const TAB = { id: "t", chatId: "c", config: {}, workspaceItem: null, projectSlug: null, appMode: "chat" } as unknown as Tab
const noop = () => {}

afterEach(cleanup)

describe("TabContent", () => {
  it("streams tokens into the chat without re-rendering the sidebar", () => {
    render(
      <TabContext.Provider value={{ tab: TAB, isActive: true, setModeLabel: noop, updateConfig: noop, setAppMode: noop, setWorkspaceItem: noop, setHistoryFile: noop, setTitle: noop }}>
        <TabContent onOpenChat={noop} onOpenProject={noop} onProjectRenamed={noop} onDocumentRenamed={noop} onNewChat={noop} onOpenItem={noop} focus={{ qaIndex: null, key: 0 }} />
      </TabContext.Provider>,
    )
    const rendersBeforeStream = probe.sidebarRenders
    expect(rendersBeforeStream).toBeGreaterThan(0)

    let content = ""
    for (const token of ["Hel", "lo", " world"]) {
      content += token
      act(() => probe.setMessages!([{ role: "assistant", content } as Message]))
    }

    expect(screen.getByTestId("chat").textContent).toBe("Hello world")
    expect(probe.sidebarRenders).toBe(rendersBeforeStream)
  })
})
