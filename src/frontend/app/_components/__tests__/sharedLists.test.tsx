import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { createRef } from "react"
import { act, cleanup, fireEvent, render, waitFor, within, type RenderResult } from "@testing-library/react"
import { Sidebar } from "@/features/navigation/Sidebar"
import { nextTab } from "@/features/navigation/tabs"
import { useProjectDocuments } from "@/features/chat/useProjectDocuments"
import type { ChatInputHandle } from "@/features/chat/composer/ChatInput"
import { TabContext, type Tab } from "@/contexts/TabContext"
import { watchDocumentLists } from "@/lib/backend/resources"
import type { ChangeEvent } from "@/lib/backend/syncStream"
import type { FileTreeNode } from "@/lib/domain/files"
import type { ProjectDocument } from "@/lib/domain/projects"

/** The file mount tree as the backend answers it. */
type RootTree = { enabled: boolean; tree: FileTreeNode[] }

const fixtures = vi.hoisted(() => ({ listeners: new Set<(event: ChangeEvent) => void>() }))
const api = vi.hoisted(() => ({
  listNotes: vi.fn(async () => []),
  listProjectDocuments: vi.fn<(slug: string, signal?: AbortSignal) => Promise<ProjectDocument[]>>(async () => []),
  listAllDocuments: vi.fn(async () => []),
  loadTemplateSets: vi.fn(async () => [{ name: "deck-set", label: "Deck" }]),
  fileRootTree: vi.fn<() => Promise<RootTree>>(),
  fetchWorktrees: vi.fn(async () => ({ is_clone: false, default_base: null, branches: [], worktrees: [] })),
  fetchOmpArchive: vi.fn(async () => []),
  fetchFactoryRepos: vi.fn(async () => []),
}))
vi.mock("@/lib/backend/api", async (importOriginal) => ({ ...await importOriginal<typeof import("@/lib/backend/api")>(), ...api }))
vi.mock("@/lib/backend/syncStream", () => ({
  subscribeToChanges: (listener: (event: ChangeEvent) => void) => {
    fixtures.listeners.add(listener)
    return () => { fixtures.listeners.delete(listener) }
  },
  syncCursorReady: async () => {},
}))
vi.mock("@/contexts/SettingsContext", async (importOriginal) => ({ ...await importOriginal<typeof import("@/contexts/SettingsContext")>(), useHostAccess: () => false }))
vi.mock("@/contexts/UndoDeleteContext", () => ({ useUndoDelete: () => ({ schedule: vi.fn(), pending: () => false }) }))
vi.mock("@/features/navigation/SidebarContext", () => ({ useSidebar: () => ({ collapsed: false, toggleCollapsed: vi.fn() }) }))
vi.mock("@/features/navigation/BranchContext", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/features/navigation/BranchContext")>(),
  useBranches: () => ({ branchMeta: {}, visibleRootFiles: [], historiesLoading: false, refreshAll: async () => {}, moveHistory: vi.fn() }),
}))
vi.mock("@/contexts/ProjectContext", async () => {
  const { displayName } = await import("@/lib/domain/projects")
  const projects = [{ slug: "proj", name: "Proj", tabs: [] }, { slug: "other", name: "Other", tabs: [] }]
  return {
    useProject: () => ({ projects, createProject: vi.fn(), deleteProject: vi.fn(), renameProject: vi.fn() }),
    useRenameDocument: () => vi.fn(),
    useDisplayName: () => (slug: string | null, path: string) => displayName(projects, slug, path),
  }
})
vi.mock("@/features/memory/MemoryViewerContext", () => ({ useMemoryViewer: () => ({ openMemoryViewer: vi.fn() }) }))
vi.mock("@/features/omp/OmpProvider", () => ({ useOmp: () => ({ overview: null, refreshOverview: () => {} }) }))
vi.mock("@/features/learn/LearnProvider", () => ({
  useLearn: () => ({
    sessions: { sessions: [], error: null, refresh: async () => {}, claim: async () => {}, rename: vi.fn(), remove: vi.fn(), retitle: vi.fn(), missing: new Set<string>() },
    sendToLearnFrame: vi.fn(), showInFrame: vi.fn(),
  }),
  useLearnDrive: () => "matched",
  useLearnFrameState: () => ({}),
  useLearnFrameEvent: () => {},
}))

const TREE: RootTree = { enabled: true, tree: [{ path: "/source", name: "source", type: "root", children: [] }] }

/** The chat's document picker of one tab, opened with its button so it reads all documents. */
function Picker({ isActive }: { isActive: boolean }) {
  const { openDocuments } = useProjectDocuments({
    isActive, appMode: "chat", projectSlug: "proj", chatId: "chat", chatInputRef: createRef<ChatInputHandle>(), track: (work) => work(),
  })
  return <button onClick={openDocuments}>Open picker</button>
}

/** One mounted tab: its sidebar with the workbench available, so it reads the template sets, and its document picker. */
function TabView({ tab, isActive }: { tab: Tab; isActive: boolean }) {
  return <TabContext.Provider value={{ tab, isActive, setModeLabel: vi.fn(), updateConfig: vi.fn(), setAppMode: vi.fn(), setWorkspaceItem: vi.fn(), setHistoryFile: vi.fn(), setTitle: vi.fn() }}>
    <Sidebar appMode="chat" onWorkbenchSelect={vi.fn()} />
    <Picker isActive={isActive} />
  </TabContext.Provider>
}

/** Open every tab's document picker and wait until every list has loaded. */
async function openPickersAndLoad(view: RenderResult) {
  for (const button of view.getAllByRole("button", { name: "Open picker" })) fireEvent.click(button)
  await waitFor(() => {
    for (const read of [api.listNotes, api.listAllDocuments, api.loadTemplateSets, api.fileRootTree, api.fetchWorktrees]) expect(read).toHaveBeenCalled()
    expect(api.listProjectDocuments).toHaveBeenCalledTimes(2)
  })
  await act(async () => {})
}

/** Another device's change event for `key`. */
const emit = (key: string, operation: ChangeEvent["operation"]) => act(() => {
  for (const listener of fixtures.listeners) listener({ seq: 1, resource_kind: "document", resource_key: key, version: null, operation, device_id: "other" })
})

/** The app shell's one subscription that keeps the shared lists current. */
let stopWatching = () => {}

beforeEach(() => {
  const stored = new Map<string, string>()
  vi.stubGlobal("localStorage", { getItem: (key: string) => stored.get(key) ?? null, setItem: (key: string, value: string) => { stored.set(key, value) } })
  api.fileRootTree.mockResolvedValue(TREE)
  api.listProjectDocuments.mockImplementation(async () => [])
  stopWatching = watchDocumentLists()
})
afterEach(() => {
  stopWatching()
  cleanup()
  vi.unstubAllGlobals()
  fixtures.listeners.clear()
  vi.clearAllMocks()
})

describe("the sidebar's shared lists", () => {
  it("read nothing with change events when a tab becomes active, and the mounts once while the shown tree stays", async () => {
    const tab = nextTab()
    const view = render(<TabView tab={tab} isActive />)
    await openPickersAndLoad(view)
    vi.clearAllMocks()
    const { promise: nextTree, resolve: answerTree } = Promise.withResolvers<RootTree>()
    api.fileRootTree.mockReturnValue(nextTree)

    view.rerender(<TabView tab={tab} isActive={false} />)
    view.rerender(<TabView tab={tab} isActive />)

    await waitFor(() => expect(api.fileRootTree).toHaveBeenCalledOnce())
    expect(within(view.getByLabelText("Code")).getByRole("button", { name: "source" })).toBeInTheDocument()
    await act(async () => answerTree(TREE))
    expect(api.fetchWorktrees).toHaveBeenCalledExactlyOnceWith("/source")
    for (const read of [api.listNotes, api.listProjectDocuments, api.listAllDocuments, api.loadTemplateSets]) expect(read).not.toHaveBeenCalled()
  })

  it("read each list once for two mounted tabs", async () => {
    const view = render(<><TabView tab={nextTab()} isActive /><TabView tab={nextTab()} isActive={false} /></>)
    await openPickersAndLoad(view)

    for (const read of [api.listNotes, api.listAllDocuments, api.loadTemplateSets, api.fileRootTree, api.fetchWorktrees]) expect(read).toHaveBeenCalledOnce()
    expect(api.listProjectDocuments.mock.calls.map(([slug]) => slug).sort()).toEqual(["other", "proj"])
  })

  it("read a project's documents once when another device changes its file list", async () => {
    const view = render(<><TabView tab={nextTab()} isActive /><TabView tab={nextTab()} isActive={false} /></>)
    await openPickersAndLoad(view)
    vi.clearAllMocks()

    emit("chat/proj/project.json", "write")

    await waitFor(() => expect(api.listProjectDocuments).toHaveBeenCalledExactlyOnceWith("proj", expect.any(AbortSignal)))
    expect(api.listAllDocuments).toHaveBeenCalledOnce()
    expect(api.listNotes).not.toHaveBeenCalled()
  })

  it("read a project's list holding a folder entry again when a tab becomes active, and leave a list without one", async () => {
    const folder: ProjectDocument = {
      path: "/data/workbench/latex/proj/paper", name: "paper", mime: "inode/directory",
      children: [{ path: "/data/workbench/latex/proj/paper/main.tex", name: "main.tex", type: "file" }],
    }
    api.listProjectDocuments.mockImplementation(async (slug) => slug === "proj" ? [folder] : [])
    const tab = nextTab()
    const view = render(<TabView tab={tab} isActive />)
    await openPickersAndLoad(view)
    vi.clearAllMocks()

    view.rerender(<TabView tab={tab} isActive={false} />)
    view.rerender(<TabView tab={tab} isActive />)

    await waitFor(() => expect(api.listProjectDocuments).toHaveBeenCalledExactlyOnceWith("proj", expect.any(AbortSignal)))
    await act(async () => {})
    expect(api.listProjectDocuments).not.toHaveBeenCalledWith("other", expect.anything())
  })
})
