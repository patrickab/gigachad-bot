import { afterEach, describe, expect, it, vi } from "vitest"
import type { ReactNode } from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { TabContext, type Tab } from "@/contexts/TabContext"
import type { PiRepository } from "@/lib/backend/api"
import type { FileTreeNode } from "@/lib/domain/files"
import { TabContent } from "../TabContent"

const fixtures = vi.hoisted(() => ({
  repositories: [] as PiRepository[],
  tree: [] as FileTreeNode[],
  handleSend: vi.fn(),
}))

vi.mock("@/contexts/SettingsContext", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/contexts/SettingsContext")>(),
  useHostAccess: () => true,
}))
vi.mock("@/features/pi/sessionStore", () => ({
  usePiSessions: () => ({ sessions: [], loading: false, error: null }),
  usePiRepositories: () => fixtures.repositories,
}))
vi.mock("@/lib/backend/api", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/lib/backend/api")>(),
  fetchPiRepositories: async () => fixtures.repositories,
  fileRootTree: async () => ({ enabled: true, tree: fixtures.tree }),
}))
// The composer's leading slot and notice, and a send button standing in for the input.
vi.mock("@/features/chat/ChatContainer", () => ({
  ChatContainer: ({ composerLeading, composerNotice, onSend }: { composerLeading?: ReactNode; composerNotice?: string; onSend: (text: string, attachments: []) => void }) => <div>
    {composerLeading}
    {composerNotice && <p role="alert">{composerNotice}</p>}
    <button onClick={() => onSend("hello", [])}>Send</button>
  </div>,
}))
vi.mock("@/features/navigation/Sidebar", () => ({ Sidebar: () => null }))
vi.mock("../WorkspaceSurface", () => ({ WorkspaceSurface: ({ chatBody }: { chatBody: ReactNode }) => <>{chatBody}</> }))
vi.mock("../MemoryCommandLayer", () => ({ MemoryCommandLayer: () => null }))
vi.mock("@/features/navigation/TabPicker", () => ({ TabPicker: () => null }))
vi.mock("@/features/editor/FilePanel", () => ({ FilePanelSlot: () => null }))
vi.mock("@/features/settings/PromptEditor", () => ({ PromptEditor: () => null }))
vi.mock("@/components/ui/SaveAsModal", () => ({ SaveAsModal: () => null }))
vi.mock("@/features/navigation/SidebarContext", () => ({ useSidebar: () => ({ collapsed: false, toggleCollapsed: () => {} }) }))
vi.mock("@/features/chat/EnabledToolsContext", () => ({ useEnabledTools: () => ({ researchEnabled: false }) }))
vi.mock("@/hooks/useResource", () => ({ useResource: () => undefined }))
vi.mock("../useTabChat", () => ({
  useTabChat: () => ({
    chat: { messages: [], isStreaming: false, models: null, totalUsage: undefined },
    historyLoading: false,
    filePicker: { standaloneFile: null, rootFiles: { enabled: false } },
    docs: {}, persistence: {}, turns: { handleSend: fixtures.handleSend }, memory: {}, menuItems: [],
  }),
}))

const TAB = { id: "t", chatId: "c", config: {}, workspaceItem: null, projectSlug: "proj", appMode: "chat" } as unknown as Tab
const noop = () => {}

afterEach(() => {
  cleanup()
  fixtures.repositories = []
  fixtures.tree = []
  fixtures.handleSend.mockReset()
})

function renderTab() {
  const onOpenItem = vi.fn()
  render(
    <TabContext.Provider value={{ tab: TAB, isActive: true, setModeLabel: noop, updateConfig: noop, setAppMode: noop, setWorkspaceItem: noop, setHistoryFile: noop, setTitle: noop }}>
      <TabContent onOpenChat={noop} onOpenProject={noop} onProjectRenamed={noop} onDocumentRenamed={noop} onNewChat={noop} onOpenItem={onOpenItem} focus={{ qaIndex: null, key: 0 }} />
    </TabContext.Provider>,
  )
  return onOpenItem
}

async function pick(name: string) {
  fireEvent.click(await screen.findByRole("button", { name: /^Project: / }))
  const unused = screen.queryByRole("option", { name: /Not yet used/ })
  if (unused && !screen.queryByRole("option", { name })) fireEvent.click(unused)
  fireEvent.click(await screen.findByRole("option", { name }))
}

describe("the empty chat's project selector", () => {
  it("starts the first message in a pi tab of the picked Pi checkout", async () => {
    fixtures.repositories = [{ name: "repo", path: "/home/u/git/repo" }]
    const onOpenItem = renderTab()
    await pick("repo")
    await waitFor(() => expect(screen.getByRole("button", { name: "Project: repo" })).toBeVisible())

    fireEvent.click(screen.getByRole("button", { name: "Send" }))

    expect(onOpenItem).toHaveBeenCalledWith({ kind: "pi", path: "repo", projectSlug: "proj", pi: { sessionId: null, repo: "repo" } })
    expect(fixtures.handleSend).not.toHaveBeenCalled()
  })

  it("refuses a mounted folder outside ~/git and keeps the message in the chat", async () => {
    fixtures.repositories = [{ name: "repo", path: "/home/u/git/repo" }]
    fixtures.tree = [{ path: "/mnt/notes", name: "notes", type: "root", children: [] }]
    const onOpenItem = renderTab()
    await pick("notes")

    expect(await screen.findByRole("alert")).toHaveTextContent("notes is not a Git checkout under ~/git, Pi sessions start only there")
    fireEvent.click(screen.getByRole("button", { name: "Send" }))
    expect(onOpenItem).not.toHaveBeenCalled()
    expect(fixtures.handleSend).toHaveBeenCalledWith("hello", [])
  })
})
