import { afterEach, beforeEach, describe, it, expect, vi } from "vitest"
import { createRef, type ReactElement } from "react"
import { act, cleanup, render as testingRender, fireEvent, waitFor, within } from "@testing-library/react"
import { Sidebar } from "@/features/navigation/Sidebar"
import { TabManager, TabSwitcher, nextTab, type TabManagerHandle } from "@/features/navigation/TabManager"
import { TabContext, useTab } from "@/contexts/TabContext"
import { BranchProvider, useBranches } from "@/features/navigation/BranchContext"
import type { BranchMeta } from "@/lib/domain/types"

const render = (element: ReactElement) => testingRender(
  <TabContext.Provider value={{ tab: nextTab(), isActive: true, setModeLabel: vi.fn(), updateConfig: vi.fn(), setAppMode: vi.fn(), setWorkspaceItem: vi.fn(), setHistoryFile: vi.fn(), setTitle: vi.fn() }}>
    {element}
  </TabContext.Provider>,
)

const fixtures = vi.hoisted(() => ({
  projects: [{ slug: "proj", name: "Proj", tabs: [] }, { slug: "empty", name: "Empty project", tabs: [] }],
  documents: [
    { path: "project/proj/document/board.canvas", name: "board.canvas", mime: "text/plain" },
    { path: "graph/proj/backend/backend.c4", name: "backend/backend.c4", mime: "text/plain" },
    { path: "project/proj/document/notes.md", name: "notes.md", mime: "text/markdown" },
  ],
}))
const api = vi.hoisted(() => ({
  listNotes: vi.fn(async () => [{ path: "chat_histories/_notes/loose-board.canvas", name: "loose-board.canvas", mime: "text/plain" }]),
  listProjectDocuments: vi.fn(async (slug: string) => slug === "proj" ? fixtures.documents : []),
  listChatHistories: vi.fn<typeof import("@/lib/backend/api").listChatHistories>(async () => ({ files: [] })),
  fetchBranchMeta: vi.fn<typeof import("@/lib/backend/api").fetchBranchMeta>(async () => ({})),
  moveHistoryItem: vi.fn<typeof import("@/lib/backend/api").moveHistoryItem>(),
  writeDocument: vi.fn<typeof import("@/lib/backend/api").writeDocument>(async (scope, name) => ({
    path: `project/${scope}/document/${name}`, name, mime: "text/plain",
  })),
  addRootFilesToProject: vi.fn<typeof import("@/lib/backend/api").addRootFilesToProject>(async () => []),
  fileRootTree: vi.fn<typeof import("@/lib/backend/api").fileRootTree>(async () => ({ enabled: true, tree: [{
    path: "/source", name: "source", type: "root", children: [
      { path: "/source/first.md", name: "first.md", type: "file" },
      { path: "/source/second.md", name: "second.md", type: "file" },
      { path: "/source/docs", name: "docs", type: "folder", children: [{ path: "/source/docs/guide.md", name: "guide.md", type: "file" }] },
    ],
  }] })),
}))
vi.mock("@/lib/backend/api", () => ({
  ...api,
  addFileRoot: vi.fn(), createArchitectureSource: vi.fn(), fetchProviderUsage: vi.fn(async () => ({ reports: [] })),
  removeFileRoot: vi.fn(), moveDocument: vi.fn(), removeDocument: vi.fn(),
}))
vi.mock("@/features/navigation/SidebarContext", () => ({ useSidebar: () => ({ collapsed: false }) }))
vi.mock("@/features/navigation/BranchContext", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/features/navigation/BranchContext")>(),
  useBranches: vi.fn(() => ({
    branchMeta: {}, visibleRootFiles: [], historiesLoading: false, refreshAll: vi.fn(async () => {}), moveHistory: vi.fn(),
  })),
}))
vi.mock("@/contexts/ProjectContext", () => ({
  useProject: () => ({
    projects: fixtures.projects,
    createProject: vi.fn(), deleteProject: vi.fn(),
  }),
}))
vi.mock("@/features/memory/MemoryViewerContext", () => ({ useMemoryViewer: () => ({ openMemoryViewer: vi.fn() }) }))
vi.mock("@/contexts/UndoDeleteContext", () => ({ useUndoDelete: () => ({ schedule: vi.fn(), pending: () => false }) }))

beforeEach(() => {
  const stored = new Map<string, string>()
  vi.stubGlobal("localStorage", {
    getItem: (key: string) => stored.get(key) ?? null,
    setItem: (key: string, value: string) => { stored.set(key, value) },
  })
  vi.resetAllMocks()
})
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

function WorkspaceLabel() {
  const { tab } = useTab()
  return <output>{tab.name ?? "Initial workspace"}</output>
}

function WorkspaceDraft() {
  const { tab } = useTab()
  return <div data-workspace-surface><input aria-label={`Draft ${tab.name ?? "Initial"}`} defaultValue="" /></div>
}

describe("Project navigation", () => {
  it("opens immediate categories by default and lists mounted roots collapsed under Workspaces", async () => {
    localStorage.setItem("expanded_workspace_projects", "[]")
    const view = render(<Sidebar appMode="chat" />)
    const general = within(view.getByLabelText("General"))
    expect(general.getByRole("button", { name: "canvases" })).toHaveAttribute("aria-expanded", "true")
    expect(general.queryByRole("button", { name: "files" })).not.toBeInTheDocument()
    const workspaces = within(view.getByLabelText("Workspaces"))
    expect(await workspaces.findByRole("button", { name: "source" })).toHaveAttribute("aria-expanded", "false")
    expect(workspaces.queryByRole("button", { name: "first.md" })).not.toBeInTheDocument()

    fireEvent.click(view.getByRole("button", { name: "Expand Proj" }))
    const project = within(view.getByLabelText("Proj contents"))
    for (const category of ["sessions", "canvases", "files"]) {
      expect(project.getByRole("button", { name: category })).toHaveAttribute("aria-expanded", "true")
    }
    expect(project.queryByRole("button", { name: "architecture" })).not.toBeInTheDocument()
    expect(await project.findByRole("button", { name: "board" })).toBeVisible()
    expect(project.getByRole("button", { name: "notes.md" })).toBeVisible()
    fireEvent.click(project.getByRole("button", { name: "canvases" }))
    fireEvent.click(view.getByRole("button", { name: "Collapse Proj" }))
    fireEvent.click(view.getByRole("button", { name: "Expand Proj" }))
    expect(within(view.getByLabelText("Proj contents")).getByRole("button", { name: "canvases" })).toHaveAttribute("aria-expanded", "false")
    view.unmount()
    const reopened = render(<Sidebar appMode="chat" />)
    expect(within(reopened.getByLabelText("Proj contents")).getByRole("button", { name: "canvases" })).toHaveAttribute("aria-expanded", "false")
  })

  it("reopens a workspace with its contents collapsed, ignoring saved expansion", async () => {
    localStorage.setItem("expanded_workspace_projects", JSON.stringify({ "/source": true }))
    const view = render(<Sidebar appMode="chat" />)
    const workspaces = within(view.getByLabelText("Workspaces"))
    const source = await workspaces.findByRole("button", { name: "source" })
    expect(source).toHaveAttribute("aria-expanded", "false")
    fireEvent.click(source)
    fireEvent.click(workspaces.getByRole("button", { name: "docs" }))
    expect(workspaces.getByRole("button", { name: "guide.md" })).toBeVisible()
    fireEvent.click(source)
    fireEvent.click(source)
    expect(workspaces.getByRole("button", { name: "docs" })).toHaveAttribute("aria-expanded", "false")
    expect(workspaces.queryByRole("button", { name: "guide.md" })).not.toBeInTheDocument()
  })

  it("opens canvases, architecture and files from chat without mixing their categories", async () => {
    const onCanvasSelect = vi.fn()
    const onArchitectureSelect = vi.fn()
    const onDocumentSelect = vi.fn()
    const onProjectSelect = vi.fn()
    const view = render(<Sidebar appMode="chat" onProjectSelect={onProjectSelect}
      onCanvasSelect={onCanvasSelect} onArchitectureSelect={onArchitectureSelect} onDocumentSelect={onDocumentSelect} />)

    fireEvent.click(view.getByRole("button", { name: "Proj" }))
    const project = within(view.getByLabelText("Proj contents"))
    fireEvent.click(await project.findByText("board"))
    expect(onCanvasSelect).toHaveBeenLastCalledWith("project/proj/document/board.canvas", "proj")
    fireEvent.click(view.getByRole("button", { name: "Open architecture for Proj" }))
    expect(onArchitectureSelect).toHaveBeenLastCalledWith("proj")
    fireEvent.click(await project.findByText("notes.md"))
    expect(onDocumentSelect).toHaveBeenLastCalledWith("project/proj/document/notes.md", "proj")
    expect(project.queryByText("board.canvas")).not.toBeInTheDocument()
    expect(project.queryByText("backend/backend.c4")).not.toBeInTheDocument()
    expect(onProjectSelect).toHaveBeenCalledTimes(1)
  })

  it("creates the architecture of a project without one when its icon is used", async () => {
    const onArchitectureSelect = vi.fn()
    const view = render(<Sidebar appMode="chat" onArchitectureSelect={onArchitectureSelect} />)

    fireEvent.click(await view.findByRole("button", { name: "Create architecture for Empty project" }))

    await waitFor(() => expect(onArchitectureSelect).toHaveBeenCalledWith("empty"))
    expect(vi.mocked((await import("@/lib/backend/api")).createArchitectureSource)).toHaveBeenCalledWith("empty", "system.c4")
  })

  it("keeps expansion independent from project selection and shares it across mounted views", async () => {
    const onProjectSelect = vi.fn()
    const first = render(<Sidebar appMode="chat" onProjectSelect={onProjectSelect} />)
    fireEvent.click(first.getByRole("button", { name: "Expand Proj" }))
    expect(onProjectSelect).not.toHaveBeenCalled()
    const project = within(first.getByLabelText("Proj contents"))
    await project.findByText("board")

    const second = within(render(<Sidebar appMode="canvas" onProjectSelect={onProjectSelect} />).container)
    const secondProject = within(second.getByLabelText("Proj contents"))
    await secondProject.findByText("board")
    fireEvent.click(project.getByRole("button", { name: "canvases" }))
    await waitFor(() => expect(secondProject.queryByText("board")).not.toBeInTheDocument())
    fireEvent.click(second.getByRole("button", { name: "Proj" }))
    expect(onProjectSelect).toHaveBeenCalledTimes(1)
    fireEvent.click(second.getByRole("button", { name: "Collapse Proj" }))
    expect(onProjectSelect).toHaveBeenCalledTimes(1)
  })

  it("keeps loose canvases accessible and lets empty projects create their first item", async () => {
    const onCanvasSelect = vi.fn()
    const onNewChat = vi.fn()
    const view = render(<Sidebar appMode="canvas" onCanvasSelect={onCanvasSelect} onNewChat={onNewChat} />)
    fireEvent.click(view.getByRole("button", { name: "Empty project" }))
    const empty = within(view.getByLabelText("Empty project contents"))
    fireEvent.click(empty.getByRole("button", { name: "New chat in Empty project" }))
    expect(onNewChat).toHaveBeenLastCalledWith("empty")

    const general = within(view.getByLabelText("General"))
    fireEvent.click(await general.findByText("loose-board"))
    expect(onCanvasSelect).toHaveBeenLastCalledWith("chat_histories/_notes/loose-board.canvas", "")
  })

  it("keeps a failed creation editable and navigates only after a successful retry", async () => {
    api.writeDocument.mockRejectedValueOnce(new Error("Storage unavailable"))
    const onCanvasSelect = vi.fn()
    const view = render(<Sidebar appMode="chat" onCanvasSelect={onCanvasSelect} />)
    fireEvent.click(view.getByRole("button", { name: "Expand Proj" }))
    const project = within(view.getByLabelText("Proj contents"))
    fireEvent.click(project.getByRole("button", { name: "New canvas in Proj" }))
    const input = project.getByRole("textbox", { name: "Canvas name" })
    fireEvent.change(input, { target: { value: "draft" } })
    fireEvent.keyDown(input, { key: "Enter" })
    expect(await project.findByRole("alert")).toHaveTextContent("Storage unavailable")
    expect(input).toHaveValue("draft")
    expect(onCanvasSelect).not.toHaveBeenCalled()
    fireEvent.keyDown(input, { key: "Enter" })
    await waitFor(() => expect(project.queryByRole("textbox", { name: "Canvas name" })).not.toBeInTheDocument())
    expect(onCanvasSelect).toHaveBeenCalledTimes(1)
    expect(onCanvasSelect).toHaveBeenCalledWith("project/proj/document/draft.canvas", "proj")
  })

  it("imports a multi-file selection into the targeted project's files category", async () => {
    const importedDocument = { path: "project/proj/document/imported.md", name: "imported.md", mime: "text/markdown" }
    let imported = false
    api.listProjectDocuments.mockImplementation(async (slug) => slug === "proj"
      ? imported ? [...fixtures.documents, importedDocument] : fixtures.documents
      : [])
    api.addRootFilesToProject.mockImplementation(async () => {
      imported = true
      return [importedDocument]
    })
    const view = render(<Sidebar appMode="chat" />)
    fireEvent.click(view.getByRole("button", { name: "Expand Proj" }))
    const project = within(view.getByLabelText("Proj contents"))
    const shared = within(view.getByLabelText("Workspaces"))
    fireEvent.click(await shared.findByRole("button", { name: "source" }))
    const first = shared.getByRole("button", { name: "first.md" })
    const second = shared.getByRole("button", { name: "second.md" })
    fireEvent.click(first, { ctrlKey: true })
    fireEvent.click(second, { shiftKey: true })
    const payload = new Map<string, string>()
    const dataTransfer = {
      get types() { return [...payload.keys()] },
      setData: (type: string, value: string) => { payload.set(type, value) },
      getData: (type: string) => payload.get(type) ?? "",
    }
    fireEvent.dragStart(first, { dataTransfer })
    fireEvent.dragOver(project.getByRole("button", { name: "files" }), { dataTransfer })
    fireEvent.drop(project.getByRole("button", { name: "files" }), { dataTransfer })
    expect(await project.findByRole("button", { name: "imported.md" })).toBeInTheDocument()
    expect(api.addRootFilesToProject).toHaveBeenCalledWith("proj", ["/source/first.md", "/source/second.md"])
  })

  it("keeps keyboard traversal within the tree from switching workspaces", () => {
    const manager = createRef<TabManagerHandle>()
    const view = render(<TabManager ref={manager}><Sidebar appMode="chat" /><WorkspaceLabel /></TabManager>)
    act(() => manager.current!.openTab({ ...nextTab(), name: "Second workspace" }))
    fireEvent.click(view.getByRole("button", { name: "Expand Proj" }))
    const project = within(view.getAllByLabelText("Proj contents").at(-1)!)
    const category = project.getByRole("button", { name: "canvases" })
    fireEvent.keyDown(category, { key: "Tab" })
    expect(view.getByText("Second workspace")).toBeVisible()
    fireEvent.keyDown(category, { key: "Tab", shiftKey: true })
    expect(view.getByText("Second workspace")).toBeVisible()
  })

  it("moves an always-visible general history onto a collapsed project and relocates its open branches", async () => {
    const actual = await vi.importActual<typeof import("@/features/navigation/BranchContext")>("@/features/navigation/BranchContext")
    vi.mocked(useBranches).mockImplementation(actual.useBranches)
    const root: BranchMeta = { chat_id: "root", parent_id: null, branch_message_idx: null, children: [{ chat_id: "child", branch_message_idx: 0 }], qa_count: 0 }
    const child: BranchMeta = { chat_id: "child", parent_id: "root", branch_message_idx: 0, children: [], qa_count: 0 }
    api.listChatHistories.mockResolvedValue({ files: ["loose.json", "child.json"] })
    api.fetchBranchMeta.mockResolvedValue({ "loose.json": root, "child.json": child })
    api.moveHistoryItem.mockImplementation(async () => {
      api.listChatHistories.mockResolvedValue({ files: [] })
      api.fetchBranchMeta.mockResolvedValue({ "proj/loose-2.json": root, "proj/child-2.json": child })
      return { status: "ok", new_path: "proj/loose-2.json" }
    })
    const manager = createRef<TabManagerHandle>()
    const view = render(<BranchProvider onFileClick={vi.fn()} onHistoriesMoved={(moves) => manager.current!.relocateHistories(moves)}>
      <TabManager ref={manager}><Sidebar appMode="chat" /><TabSwitcher /><WorkspaceDraft /></TabManager>
    </BranchProvider>)
    const rootTab = { ...nextTab(), name: "Root", historyFile: "loose.json" }
    const childTab = { ...nextTab(), name: "Branch", historyFile: "child.json" }
    act(() => {
      manager.current!.openTab(rootTab)
      manager.current!.openTab(childTab)
    })
    fireEvent.change(view.getByRole("textbox", { name: "Draft Branch" }), { target: { value: "Keep this draft" } })
    const navigation = within(view.getByRole("navigation", { name: "Workspace" }))
    const general = within(navigation.getByLabelText("General"))
    const history = await general.findByRole("button", { name: "loose" })
    expect(general.queryByRole("button", { name: "child" })).not.toBeInTheDocument()
    expect(navigation.queryByLabelText("Proj contents")).not.toBeInTheDocument()
    const payload = new Map<string, string>()
    const dataTransfer = {
      get types() { return [...payload.keys()] },
      setData: (type: string, value: string) => { payload.set(type, value) },
      getData: (type: string) => payload.get(type) ?? "",
    }
    fireEvent.dragStart(history, { dataTransfer })
    fireEvent.dragOver(navigation.getByRole("button", { name: "Proj" }), { dataTransfer })
    fireEvent.drop(navigation.getByRole("button", { name: "Proj" }), { dataTransfer })
    await waitFor(() => expect(general.queryByRole("button", { name: "loose" })).not.toBeInTheDocument())
    fireEvent.click(navigation.getByRole("button", { name: "Expand Proj" }))
    const project = within(navigation.getByLabelText("Proj contents"))
    expect(project.getByRole("button", { name: "loose-2" })).toBeVisible()
    expect(manager.current!.getTabs().find((tab) => tab.id === rootTab.id)).toMatchObject({ historyFile: "proj/loose-2.json", projectSlug: "proj" })
    expect(manager.current!.getTabs().find((tab) => tab.id === childTab.id)).toMatchObject({ historyFile: "proj/child-2.json", projectSlug: "proj" })
    expect(view.getByRole("textbox", { name: "Draft Branch" })).toHaveValue("Keep this draft")
    expect(api.moveHistoryItem).toHaveBeenCalledWith("loose.json", "proj")
  })

  it("keeps a general history visible when dropping it into a project fails", async () => {
    const actual = await vi.importActual<typeof import("@/features/navigation/BranchContext")>("@/features/navigation/BranchContext")
    vi.mocked(useBranches).mockImplementation(actual.useBranches)
    api.listChatHistories.mockResolvedValue({ files: ["loose.json"] })
    api.moveHistoryItem.mockRejectedValue(new Error("Storage unavailable"))
    const view = render(<BranchProvider onFileClick={vi.fn()}><Sidebar appMode="chat" /></BranchProvider>)
    const general = within(view.getByLabelText("General"))
    const history = await general.findByRole("button", { name: "loose" })
    const dataTransfer = { types: ["text/plain"], getData: (type: string) => type === "text/plain" ? "loose.json" : "" }
    fireEvent.drop(view.getByRole("button", { name: "Proj" }), { dataTransfer })
    expect(await view.findByRole("alert")).toHaveTextContent("Storage unavailable")
    expect(history).toBeVisible()
    expect(view.queryByLabelText("Proj contents")).not.toBeInTheDocument()
  })
})

describe("Toolbar tab switching", () => {
  it("toggles two tabs directly and disables cycling when only one remains", () => {
    const manager = createRef<TabManagerHandle>()
    const view = render(<TabManager ref={manager}><TabSwitcher /></TabManager>)
    expect(view.getByRole("button", { name: "Next tab" })).toBeDisabled()
    act(() => manager.current!.openTab({ ...nextTab(), name: "Other" }))

    for (const name of ["New chat", "Other"]) {
      fireEvent.click(view.getByRole("button", { name: "Next tab" }))
      expect(view.getByRole("button", { name: `Switch tabs: general / sessions / ${name}` })).toBeVisible()
      expect(view.queryByRole("dialog")).not.toBeInTheDocument()
    }

    fireEvent.click(view.getByRole("button", { name: /^Switch tabs:/ }))
    fireEvent.click(view.getByRole("button", { name: "Close general / sessions / Other" }))
    expect(view.getByRole("button", { name: "Next tab" })).toBeDisabled()
    expect(view.getByRole("button", { name: "Switch tabs: general / sessions / New chat" })).toBeVisible()
  })

  it("cycles in natural menu order rather than creation order and wraps at the end", () => {
    const manager = createRef<TabManagerHandle>()
    const view = render(<TabManager ref={manager}><TabSwitcher /></TabManager>)
    act(() => {
      manager.current!.openTab({ ...nextTab(), name: "Chat 10" })
      manager.current!.openTab({ ...nextTab(), name: "Chat 2" })
    })

    for (const name of ["Chat 10", "New chat", "Chat 2"]) {
      fireEvent.click(view.getByRole("button", { name: "Next tab" }))
      expect(view.getByRole("button", { name: `Switch tabs: general / sessions / ${name}` })).toBeVisible()
    }
  })

  it("reuses the selected tab for sessions and items without disturbing background tabs", () => {
    const manager = createRef<TabManagerHandle>()
    const view = render(<TabManager ref={manager}><TabSwitcher /><WorkspaceDraft /></TabManager>)
    act(() => manager.current!.showChat("proj/first.json"))
    const background = manager.current!.getTabs()[0]
    const selected = nextTab()
    act(() => manager.current!.openTab(selected))
    fireEvent.change(view.getByRole("textbox"), { target: { value: "Unsaved draft" } })

    act(() => manager.current!.showChat("proj/first.json"))
    expect(view.getByRole("button", { name: "Switch tabs: Proj / sessions / first.json" })).toBeVisible()
    expect(view.getByRole("textbox")).toHaveValue("")
    fireEvent.change(view.getByRole("textbox"), { target: { value: "Session draft" } })
    act(() => manager.current!.showItem({ kind: "canvas", path: "board.canvas", projectSlug: "proj" }))
    expect(view.getByRole("button", { name: "Switch tabs: Proj / canvases / board.canvas" })).toBeVisible()
    act(() => manager.current!.showChat("proj/first.json"))
    expect(view.getByRole("textbox")).toHaveValue("Session draft")

    act(() => manager.current!.showItem({ kind: "document", path: "/source/notes.md", projectSlug: null, rootFile: true }))
    expect(view.getByRole("button", { name: "Switch tabs: general / files / notes.md" })).toBeVisible()
    act(() => manager.current!.showChat("proj/second.json"))
    expect(view.getByRole("button", { name: "Switch tabs: Proj / sessions / second.json" })).toBeVisible()
    expect(view.getByRole("textbox")).toHaveValue("")
    expect(manager.current!.getTabs().map((tab) => tab.id)).toEqual([background.id, selected.id])
    expect(manager.current!.getTabs()[0]).toEqual(background)
  })

  it("tracks the visible surface across project, shared, and scratch workspaces", () => {
    const manager = createRef<TabManagerHandle>()
    const view = render(<TabManager ref={manager}><TabSwitcher /></TabManager>)
    const canvas = { ...nextTab(), appMode: "canvas" as const, projectSlug: "proj", workspaceItem: { kind: "canvas" as const, path: "project/proj/document/board.canvas", projectSlug: "proj" } }
    const architecture = { ...nextTab(), appMode: "canvas" as const, projectSlug: "proj", workspaceItem: { kind: "architecture" as const, path: "graph/proj/", projectSlug: "proj" } }
    const file = { ...nextTab(), appMode: "document" as const, workspaceItem: { kind: "document" as const, path: "/source/notes.md", projectSlug: null, rootFile: true } }
    const scratch = { ...nextTab(), appMode: "canvas" as const, projectSlug: "proj" }
    act(() => {
      for (const tab of [canvas, architecture, file, scratch]) manager.current!.openTab(tab)
    })
    expect(view.getByRole("button", { name: "Switch tabs: Proj / canvases / new.canvas" })).toBeVisible()
    for (const breadcrumb of ["Proj / canvases / board.canvas", "Proj / architecture / System overview", "general / files / notes.md"]) {
      fireEvent.click(view.getByRole("button", { name: /^Switch tabs:/ }))
      const dialog = within(view.getByRole("dialog", { name: "Open tabs" }))
      fireEvent.click(dialog.getByRole("button", { name: `Switch to ${breadcrumb}` }))
      expect(view.queryByRole("dialog")).not.toBeInTheDocument()
      expect(view.getByRole("button", { name: `Switch tabs: ${breadcrumb}` })).toBeVisible()
    }
    act(() => manager.current!.showChat("proj/session.json"))
    expect(view.getByRole("button", { name: "Switch tabs: Proj / sessions / session.json" })).toBeVisible()
  })

  it("keeps search usable after closing a filtered tab and preserves keyboard focus", () => {
    const manager = createRef<TabManagerHandle>()
    const view = render(<TabManager ref={manager}><TabSwitcher /></TabManager>)
    act(() => {
      for (let index = 1; index <= 8; index++) manager.current!.openTab({ ...nextTab(), name: `Chat ${index}`, projectSlug: "proj" })
    })
    const trigger = view.getByRole("button", { name: "Switch tabs: Proj / sessions / Chat 8" })
    fireEvent.click(trigger)
    const input = view.getByRole("textbox", { name: "Search open tabs" })
    fireEvent.change(input, { target: { value: "Chat 1" } })
    const dialog = within(view.getByRole("dialog", { name: "Open tabs" }))
    expect(dialog.queryByRole("button", { name: "Switch to Proj / sessions / Chat 8" })).not.toBeInTheDocument()
    fireEvent.click(dialog.getByRole("button", { name: "Close Proj / sessions / Chat 1" }))
    expect(input).toBeVisible()
    expect(input).toHaveFocus()
    expect(dialog.getByText("No matching tabs")).toBeInTheDocument()
    fireEvent.change(input, { target: { value: "" } })
    const active = dialog.getByRole("button", { name: "Switch to Proj / sessions / Chat 8" })
    expect(fireEvent.keyDown(active, { key: "Tab" })).toBe(true)
    expect(trigger).toHaveAttribute("aria-expanded", "true")
    fireEvent.keyDown(input, { key: "Escape" })
    expect(view.queryByRole("dialog")).not.toBeInTheDocument()
    expect(trigger).toHaveFocus()
    fireEvent.click(trigger)
    expect(view.queryByRole("textbox", { name: "Search open tabs" })).not.toBeInTheDocument()
    fireEvent.click(view.getByRole("button", { name: "New chat" }))
    expect(view.getByRole("button", { name: "Switch tabs: Proj / sessions / New chat" })).toBeVisible()
    expect(manager.current!.getTabs()).toHaveLength(9)
  })

  it("sorts by project, category and natural filename order without reordering on selection", () => {
    const manager = createRef<TabManagerHandle>()
    const view = render(<TabManager ref={manager}><TabSwitcher /></TabManager>)
    const tabs = [
      { ...nextTab(), projectSlug: "proj", appMode: "document" as const, workspaceItem: { kind: "document" as const, path: "alpha.md", projectSlug: "proj" } },
      { ...nextTab(), projectSlug: "proj", appMode: "canvas" as const, workspaceItem: { kind: "canvas" as const, path: "board10.canvas", projectSlug: "proj" } },
      { ...nextTab(), projectSlug: "proj", title: "Alpha" },
      { ...nextTab(), projectSlug: "empty", title: "Z last" },
      { ...nextTab(), projectSlug: "proj", appMode: "canvas" as const, workspaceItem: { kind: "canvas" as const, path: "Board2.canvas", projectSlug: "proj" } },
      { ...nextTab(), projectSlug: "proj", appMode: "canvas" as const, workspaceItem: { kind: "architecture" as const, path: "graph/proj/", projectSlug: "proj" } },
    ]
    act(() => { for (const tab of tabs) manager.current!.openTab(tab) })
    fireEvent.click(view.getByRole("button", { name: /^Switch tabs:/ }))
    const expected = [
      "Switch to Empty project / sessions / Z last",
      "Switch to Proj / architecture / System overview",
      "Switch to Proj / canvases / Board2.canvas",
      "Switch to Proj / canvases / board10.canvas",
      "Switch to Proj / files / alpha.md",
      "Switch to Proj / sessions / Alpha",
      "Switch to general / sessions / New chat",
    ]
    const labels = () => within(view.getByRole("dialog", { name: "Open tabs" })).getAllByRole("button", { name: /^Switch to / }).map((button) => button.getAttribute("aria-label"))
    expect(labels()).toEqual(expected)
    fireEvent.click(view.getByRole("button", { name: expected[3] }))
    fireEvent.click(view.getByRole("button", { name: /^Switch tabs:/ }))
    expect(labels()).toEqual(expected)
    fireEvent.click(view.getByRole("button", { name: "Close Proj / files / alpha.md" }))
    expect(labels()).toEqual(expected.filter((label) => label !== "Switch to Proj / files / alpha.md"))
  })

  it("creates chats in the active workspace's project or general scope", () => {
    const manager = createRef<TabManagerHandle>()
    const view = render(<TabManager ref={manager}><TabSwitcher /></TabManager>)
    const projectTab = { ...nextTab(), name: "Project chat", projectSlug: "proj" }
    const sharedTab = { ...nextTab(), name: "Shared chat" }
    act(() => {
      manager.current!.openTab(projectTab)
      manager.current!.openTab(sharedTab)
    })
    for (const tab of [projectTab, sharedTab]) {
      act(() => manager.current!.switchToTab(tab.id))
      expect(view.queryByRole("dialog")).not.toBeInTheDocument()
      fireEvent.click(view.getByRole("button", { name: "New chat" }))
      expect(manager.current!.getTabs().at(-1)?.projectSlug).toBe(tab.projectSlug)
      expect(view.queryByRole("dialog")).not.toBeInTheDocument()
    }
  })

  it("preserves drafts and exposes only the current workspace during interrupted transitions", () => {
    vi.stubGlobal("matchMedia", () => ({ matches: false, addEventListener: vi.fn(), removeEventListener: vi.fn() }))
    const manager = createRef<TabManagerHandle>()
    const view = render(<TabManager ref={manager}><WorkspaceDraft /></TabManager>)
    const initial = manager.current!.getTabs()[0]
    fireEvent.change(view.getByRole("textbox", { name: "Draft Initial" }), { target: { value: "Keep my draft" } })
    const second = { ...nextTab(), name: "Second" }
    const third = { ...nextTab(), name: "Third" }
    act(() => manager.current!.openTab(second))
    act(() => manager.current!.openTab(third))
    fireEvent.change(view.getByRole("textbox", { name: "Draft Third" }), { target: { value: "Another draft" } })
    act(() => manager.current!.switchToTab(initial.id))
    expect(view.getAllByRole("textbox")).toHaveLength(1)
    expect(view.getByRole("textbox", { name: "Draft Initial" })).toHaveValue("Keep my draft")
    act(() => manager.current!.switchToTab(third.id))
    expect(view.getAllByRole("textbox")).toHaveLength(1)
    expect(view.getByRole("textbox", { name: "Draft Third" })).toHaveValue("Another draft")
  })
})
