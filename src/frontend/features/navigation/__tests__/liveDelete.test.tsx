import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { createRef, type RefObject } from "react"
import { act, cleanup, fireEvent, render, waitFor, within, type RenderResult } from "@testing-library/react"
import { Sidebar } from "@/features/navigation/Sidebar"
import { TabPicker } from "@/features/navigation/TabPicker"
import { TabManager, type TabManagerHandle } from "@/features/navigation/TabManager"
import { nextTab } from "@/features/navigation/tabs"
import { showingItem, type WorkspaceItem } from "@/contexts/TabContext"
import { UndoDeleteProvider } from "@/contexts/UndoDeleteContext"
import { watchDocumentLists } from "@/lib/backend/resources"
import type { ChangeEvent } from "@/lib/backend/syncStream"

const FOLDER = "/data/workbench/latex/proj/paper"
const TREE = [
  { path: `${FOLDER}/paper.tex`, name: "paper.tex", type: "file" },
  { path: `${FOLDER}/imgs`, name: "imgs", type: "folder", children: [{ path: `${FOLDER}/imgs/figure.png`, name: "figure.png", type: "file" }] },
]
const ENTRY = { path: FOLDER, name: "paper", mime: "inode/directory", children: TREE }
const NOTE = { path: "project/proj/document/notes.md", name: "notes.md", mime: "text/markdown" }

const fixtures = vi.hoisted(() => ({ documents: [] as { path: string }[], listeners: new Set<(event: ChangeEvent) => void>() }))
const api = vi.hoisted(() => ({
  listNotes: vi.fn(async () => []),
  listProjectDocuments: vi.fn<(slug: string) => Promise<unknown[]>>(),
  removeDocument: vi.fn<(slug: string, path: string) => Promise<string[]>>(),
  fileRootTree: vi.fn(async () => ({ enabled: true, tree: [] })),
  fetchWorktrees: vi.fn(async () => ({ is_clone: false, default_base: null, branches: [], worktrees: [] })),
  callFactory: vi.fn(async () => ({ repos: [] })),
  fetchProviderUsage: vi.fn(async () => ({ reports: [] })),
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
vi.mock("@/features/navigation/SidebarContext", () => ({ useSidebar: () => ({ collapsed: false, toggleCollapsed: vi.fn() }) }))
vi.mock("@/features/navigation/BranchContext", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/features/navigation/BranchContext")>(),
  useBranches: () => ({ branchMeta: {}, visibleRootFiles: [], historiesLoading: false, refreshAll: async () => {}, moveHistory: vi.fn() }),
  useBranchActions: () => ({ refreshAll: async () => {} }),
}))
vi.mock("@/contexts/ProjectContext", async () => {
  const { displayName } = await import("@/lib/domain/projects")
  const projects = [{ slug: "proj", name: "Proj", tabs: [] }]
  return {
    useProject: () => ({ projects, createProject: vi.fn(), deleteProject: vi.fn(), renameProject: vi.fn() }),
    useRenameDocument: () => vi.fn(),
    useDisplayName: () => (slug: string | null, path: string) => displayName(projects, slug, path),
  }
})
vi.mock("@/features/memory/MemoryViewerContext", () => ({ useMemoryViewer: () => ({ openMemoryViewer: vi.fn() }) }))
vi.mock("@/features/learn/LearnProvider", () => ({
  useLearn: () => ({
    sessions: { sessions: [], error: null, refresh: async () => {}, claim: async () => {}, rename: vi.fn(), remove: vi.fn(), retitle: vi.fn(), missing: new Set<string>() },
    sendToLearnFrame: vi.fn(), showInFrame: vi.fn(),
  }),
  useLearnDrive: () => "matched",
  useLearnFrameState: () => ({}),
  useLearnFrameEvent: () => {},
}))

/** Another device's change event for `key`. */
const emit = (key: string, operation: ChangeEvent["operation"]) => act(() => {
  for (const listener of fixtures.listeners) listener({ seq: 1, resource_kind: "document", resource_key: key, version: null, operation, device_id: "other" })
})

/** One tab showing `item`, in the shape the sidebar opens it. */
const tabShowing = (item: WorkspaceItem) =>
  ({ ...nextTab(), ...showingItem(item) })

/** The app's one sidebar beside a tab manager's tabs under the real undo provider, with tabs of the paper project open and the note's tab frontmost. */
async function renderWorkspace(openTabs = true) {
  const manager = createRef<TabManagerHandle>()
  const view = render(
    <UndoDeleteProvider>
      <TabManager ref={manager} chrome={(stack) => <>
        <Sidebar workbench onItemOpen={(item) => manager.current!.showItem(item)} />
        {stack}
        <TabPicker />
      </>}>{null}</TabManager>
    </UndoDeleteProvider>,
  )
  // The tree can suspend on its first render (a lazy module that loads slowly under load): wait for it to mount.
  await waitFor(() => expect(manager.current).not.toBeNull())
  const initial = manager.current!.getTabs()[0]
  const workbench = tabShowing({ kind: "workbench", path: `${FOLDER}/paper.tex`, projectSlug: "proj", texFolder: FOLDER })
  const image = tabShowing({ kind: "document", path: `${FOLDER}/imgs/figure.png`, projectSlug: "proj" })
  const note = tabShowing({ kind: "document", path: NOTE.path, projectSlug: "proj" })
  if (openTabs) act(() => { for (const tab of [workbench, image, note]) manager.current!.openTab(tab) })
  fireEvent.click(view.getByRole("button", { name: "Expand Proj" }))
  return { manager, view, initial, workbench, image, note }
}

const shownPaths = (manager: RefObject<TabManagerHandle | null>) => manager.current!.getTabs().map((tab) => tab.workspaceItem?.path ?? null)
/** The project's rows in the sidebar. */
const sidebar = (view: RenderResult) => within(view.getByLabelText("Proj contents"))

/** The app shell's one subscription that keeps the shared lists current. */
let stopWatching = () => {}

beforeEach(() => {
  // The project's Files section open, where folder entries list
  const stored = new Map<string, string>([["expanded_workspace_projects", JSON.stringify({ "project:proj:chats": false, "project:proj:files": true })]])
  vi.stubGlobal("localStorage", { getItem: (key: string) => stored.get(key) ?? null, setItem: (key: string, value: string) => { stored.set(key, value) } })
  fixtures.documents = [ENTRY, NOTE]
  api.listProjectDocuments.mockImplementation(async () => fixtures.documents)
  api.removeDocument.mockImplementation(async (_slug, path) => {
    fixtures.documents = fixtures.documents.filter((doc) => doc.path !== path)
    return [path]
  })
  vi.useFakeTimers({ shouldAdvanceTime: true })
  stopWatching = watchDocumentLists()
})
afterEach(() => {
  stopWatching()
  cleanup()
  vi.useRealTimers()
  vi.unstubAllGlobals()
  fixtures.listeners.clear()
  vi.clearAllMocks()
})

describe("deleting a folder entry", () => {
  it("drops its row after the deferred commit, also after a tab switch, and closes the tabs showing its files and its workbench", async () => {
    const { manager, view, initial, note } = await renderWorkspace()
    await sidebar(view).findByRole("button", { name: "paper" })

    fireEvent.click(sidebar(view).getByRole("button", { name: "Delete paper" }))
    expect(sidebar(view).queryByRole("button", { name: "paper" })).not.toBeInTheDocument()
    expect(api.removeDocument).not.toHaveBeenCalled()
    expect(manager.current!.getTabs()).toHaveLength(4)

    // After a tab switch the sidebar keeps the row hidden while the delete is still undoable and hears no change event of this device's commit.
    act(() => manager.current!.switchToTab(initial.id))
    expect(sidebar(view).queryByRole("button", { name: "paper" })).not.toBeInTheDocument()
    expect(await sidebar(view).findByRole("button", { name: "notes.md" })).toBeInTheDocument()
    await act(async () => { await vi.advanceTimersByTimeAsync(6000) })

    expect(api.removeDocument).toHaveBeenCalledWith("proj", FOLDER)
    expect(sidebar(view).queryByRole("button", { name: "paper" })).not.toBeInTheDocument()
    expect(sidebar(view).getByRole("button", { name: "notes.md" })).toBeInTheDocument()
    expect(manager.current!.getTabs().map((tab) => tab.id)).toEqual([initial.id, note.id])
  })

  it("keeps the row and every tab when it is undone", async () => {
    const { manager, view } = await renderWorkspace()
    await sidebar(view).findByRole("button", { name: "paper" })

    fireEvent.click(sidebar(view).getByRole("button", { name: "Delete paper" }))
    fireEvent.click(view.getByRole("button", { name: "Undo" }))
    await act(async () => { await vi.advanceTimersByTimeAsync(6000) })

    expect(api.removeDocument).not.toHaveBeenCalled()
    expect(sidebar(view).getByRole("button", { name: "paper" })).toBeInTheDocument()
    expect(shownPaths(manager)).toEqual([null, `${FOLDER}/paper.tex`, `${FOLDER}/imgs/figure.png`, NOTE.path])
  })

  it("does the same when another device deletes it", async () => {
    const { manager, view, initial, note } = await renderWorkspace()
    await sidebar(view).findByRole("button", { name: "paper" })
    fixtures.documents = [NOTE]

    emit(FOLDER, "delete")

    expect(sidebar(view).queryByRole("button", { name: "paper" })).not.toBeInTheDocument()
    expect(manager.current!.getTabs().map((tab) => tab.id)).toEqual([initial.id, note.id])
  })

  it("is not touched by another device writing the same path", async () => {
    const { manager, view } = await renderWorkspace()
    await sidebar(view).findByRole("button", { name: "paper" })

    emit(`${FOLDER}/paper.tex`, "write")

    expect(sidebar(view).getByRole("button", { name: "paper" })).toBeInTheDocument()
    expect(manager.current!.getTabs()).toHaveLength(4)
  })
})

describe("a tab switch in the app's one sidebar", () => {
  const MOUNT = { path: "/repo", name: "repo", type: "folder", mount: "/repo", children: [] }
  const settle = () => act(async () => { await vi.advanceTimersByTimeAsync(50) })

  it("tries a failed worktree read again", async () => {
    api.fileRootTree.mockResolvedValue({ enabled: true, tree: [MOUNT] } as never)
    api.fetchWorktrees.mockRejectedValueOnce(new Error("git is busy"))
    const { manager } = await renderWorkspace(false)
    await waitFor(() => expect(api.fetchWorktrees).toHaveBeenCalledTimes(1))
    await settle()
    expect(api.fetchWorktrees).toHaveBeenCalledTimes(1)

    act(() => manager.current!.openTab(nextTab()))
    await settle()

    expect(api.fetchWorktrees).toHaveBeenCalledTimes(2)
  })

  it("leaves no focus on a row that switched tabs, so a tapped Space opens the tab picker", async () => {
    const { manager, view, initial } = await renderWorkspace()
    act(() => manager.current!.switchToTab(initial.id))
    fireEvent.click(await sidebar(view).findByRole("button", { name: "paper" }))
    const row = await sidebar(view).findByRole("button", { name: "paper.tex" })

    // The workbench tab shows the file: the row switches there
    row.focus()
    fireEvent.click(row)
    expect(view.container.querySelector(".workspace-tab:not([aria-hidden='true'])")).not.toBe(view.container.querySelector(".workspace-tab"))
    const target = document.activeElement ?? document.body
    fireEvent.keyDown(target, { key: " ", code: "Space" })
    fireEvent.keyUp(target, { key: " ", code: "Space" })

    expect(view.getByRole("listbox", { name: "Tabs" })).toBeInTheDocument()
  })
})
