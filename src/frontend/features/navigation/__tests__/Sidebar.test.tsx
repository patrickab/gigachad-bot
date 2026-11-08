import { afterEach, beforeEach, describe, it, expect, vi } from "vitest"
import { createRef, type ReactElement } from "react"
import { act, cleanup, render as testingRender, fireEvent, waitFor, within, type RenderResult } from "@testing-library/react"
import { Sidebar } from "@/features/navigation/Sidebar"
import { TabManager, type TabManagerHandle } from "@/features/navigation/TabManager"
import { TabList } from "@/features/navigation/TabList"
import { nextTab } from "@/features/navigation/tabs"
import { TabContext, useTab } from "@/contexts/TabContext"
import { onFileChange } from "@/lib/backend/fileEvents"
import { removeDocument } from "@/lib/backend/api"
import { BranchProvider, useBranches } from "@/features/navigation/BranchContext"
import type { BranchMeta } from "@/lib/domain/histories"

const render = (element: ReactElement) => testingRender(
  <TabContext.Provider value={{ tab: nextTab(), isActive: true, setModeLabel: vi.fn(), updateConfig: vi.fn(), setAppMode: vi.fn(), setWorkspaceItem: vi.fn(), setHistoryFile: vi.fn(), setTitle: vi.fn() }}>
    {element}
  </TabContext.Provider>,
)

const fixtures = vi.hoisted(() => ({
  projects: [{ slug: "proj", name: "Proj", tabs: [], file_names: { "project/proj/document/plan.md": "Roadmap.md" } }, { slug: "empty", name: "Empty project", tabs: [] }],
  renameProject: vi.fn<(slug: string, name: string) => Promise<string>>(),
  renameDocument: vi.fn<(slug: string, path: string, name: string) => Promise<{ path: string; name: string; mime: string; changed: string[] }>>(),
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
  createWorkbenchDocument: vi.fn<typeof import("@/lib/backend/api").createWorkbenchDocument>(async (slug, name, template) =>
    `/data/workbench/latex/${slug}/${name.replace(/\.tex$/, "")}/${template ? "main.tex" : name}`),
  loadTemplateSets: vi.fn<typeof import("@/lib/backend/api").loadTemplateSets>(async () => [{ name: "deck-set", label: "Deck" }]),
  fileRootTree: vi.fn<typeof import("@/lib/backend/api").fileRootTree>(async () => ({ enabled: true, tree: [{
    path: "/source", name: "source", type: "root", children: [
      { path: "/source/first.md", name: "first.md", type: "file" },
      { path: "/source/second.md", name: "second.md", type: "file" },
      { path: "/source/docs", name: "docs", type: "folder", children: [{ path: "/source/docs/guide.md", name: "guide.md", type: "file" }] },
    ],
  }] })),
  fetchWorktrees: vi.fn<typeof import("@/lib/backend/api").fetchWorktrees>(async () => ({ is_clone: false, default_base: null, branches: [], worktrees: [] })),
  removeWorktree: vi.fn<typeof import("@/lib/backend/api").removeWorktree>(async () => {}),
  deleteWorktreeBranch: vi.fn<typeof import("@/lib/backend/api").deleteWorktreeBranch>(async () => {}),
  fetchOmpArchive: vi.fn<typeof import("@/lib/backend/api").fetchOmpArchive>(async () => []),
  archiveOmpSession: vi.fn<typeof import("@/lib/backend/api").archiveOmpSession>(async () => {}),
  restoreOmpSession: vi.fn<typeof import("@/lib/backend/api").restoreOmpSession>(async () => "restored"),
  fetchFactoryRepos: vi.fn<typeof import("@/lib/backend/api").fetchFactoryRepos>(async () => []),
  exportLearnSession: vi.fn<typeof import("@/lib/backend/api").exportLearnSession>(),
  registerFactoryRepo: vi.fn<typeof import("@/lib/backend/api").registerFactoryRepo>(),
  createTexProjectFile: vi.fn<typeof import("@/lib/backend/api").createTexProjectFile>(),
  promoteToTexProject: vi.fn<typeof import("@/lib/backend/api").promoteToTexProject>(),
}))
const login = vi.hoisted(() => ({ hostAccess: false }))
vi.mock("@/contexts/SettingsContext", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/contexts/SettingsContext")>(),
  useHostAccess: () => login.hostAccess,
}))
vi.mock("@/lib/backend/api", async (importOriginal) => {
  const { ApiError, currentConnectionPool, leaveConnectionPool } = await importOriginal<typeof import("@/lib/backend/api")>()
  return {
    ...api,
    ApiError, currentConnectionPool, leaveConnectionPool,
    addFileRoot: vi.fn(), createArchitectureSource: vi.fn(), createWorktree: vi.fn(), fetchProviderUsage: vi.fn(async () => ({ reports: [] })),
    removeFileRoot: vi.fn(), moveDocument: vi.fn(), removeDocument: vi.fn(),
  }
})
const sidebarFrame = vi.hoisted(() => ({ collapsed: false, toggleCollapsed: vi.fn() }))
vi.mock("@/features/navigation/SidebarContext", () => ({ useSidebar: () => sidebarFrame }))
vi.mock("@/features/navigation/BranchContext", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/features/navigation/BranchContext")>(),
  useBranches: vi.fn(() => ({
    branchMeta: {}, visibleRootFiles: [], historiesLoading: false, refreshAll: vi.fn(async () => {}), moveHistory: vi.fn(),
  })),
}))
vi.mock("@/contexts/ProjectContext", async () => {
  const { displayName } = await import("@/lib/domain/projects")
  return {
    useProject: () => ({
      projects: fixtures.projects,
      createProject: vi.fn(), deleteProject: vi.fn(),
      renameProject: fixtures.renameProject,
    }),
    useRenameDocument: () => fixtures.renameDocument,
    useDisplayName: () => (slug: string | null, path: string) => displayName(fixtures.projects, slug, path),
  }
})
vi.mock("@/features/memory/MemoryViewerContext", () => ({ useMemoryViewer: () => ({ openMemoryViewer: vi.fn() }) }))
const undo = vi.hoisted(() => ({ schedule: vi.fn<(key: string, label: string, run: () => Promise<void>) => void>() }))
vi.mock("@/contexts/UndoDeleteContext", () => ({ useUndoDelete: () => ({ schedule: undo.schedule, pending: () => false }) }))
const omp = vi.hoisted(() => ({ overview: null as import("@/lib/domain/types").OmpOverview | null, refreshOverview: () => {} }))
vi.mock("@/features/omp/OmpProvider", () => ({ useOmp: () => omp }))
const learn = vi.hoisted(() => ({
  sessions: {
    sessions: [] as import("@/lib/domain/types").LearnSession[], error: null as string | null, refresh: vi.fn(async () => {}), claim: async () => {},
    rename: vi.fn<(sessionId: string, title: string) => Promise<void>>(), remove: vi.fn<(sessionId: string, mode?: string) => Promise<void>>(),
    retitle: vi.fn<(sessionId: string, title: string) => void>(), missing: new Set<string>() as ReadonlySet<string>,
  },
  sendToLearnFrame: vi.fn<(message: import("@gigachad/chat-ui/learn-protocol").LearnShellMessage) => boolean>(),
  showInFrame: vi.fn<(message: import("@gigachad/chat-ui/learn-protocol").LearnShellMessage, session?: { id: string; project_slug: string | null }) => void>(),
}))
type FrameMessageOf<Type extends import("@gigachad/chat-ui/learn-protocol").LearnFrameMessageType> = import("@gigachad/chat-ui/learn-protocol").LearnFrameMessageOf<Type>
const learnFrame = vi.hoisted(() => ({
  drive: "matched" as "waiting" | "matched" | "mismatch",
  metas: {} as Record<string, FrameMessageOf<"session-meta">>,
  plans: {} as Record<string, FrameMessageOf<"session-plan">>,
  activity: null as FrameMessageOf<"activity"> | null,
  chromeKeys: new Set<(message: FrameMessageOf<"chrome-key">) => void>(),
}))
vi.mock("@/features/learn/LearnProvider", async () => {
  const { useEffect } = await import("react")
  return {
    useLearn: () => learn,
    useLearnDrive: () => learnFrame.drive,
    useLearnFrameState: (type: string) => type === "session-meta" ? learnFrame.metas : type === "session-plan" ? learnFrame.plans : learnFrame.activity,
    useLearnFrameEvent: (_type: "chrome-key", listener: (message: FrameMessageOf<"chrome-key">) => void) => {
      useEffect(() => {
        learnFrame.chromeKeys.add(listener)
        return () => { learnFrame.chromeKeys.delete(listener) }
      })
    },
  }
})

/** A learn session row as `GET /api/learn/sessions` lists it. */
function learnSession(id: string, title: string, fields: Partial<import("@/lib/domain/types").LearnSession> = {}): import("@/lib/domain/types").LearnSession {
  return { id, title, kind: "teach", busy: false, compacting: false, pending_question: false, pending_problem: false, updated_at: null, project_slug: null, ...fields }
}

function sessionMeta(sessionId: string, title: string, fields: Partial<FrameMessageOf<"session-meta">> = {}): FrameMessageOf<"session-meta"> {
  return { type: "session-meta", sessionId, title, kind: "teach", phase: "probe", status: null, updatedAt: "2026-10-29T10:00:00.000Z", ...fields }
}

/** Render `element` in a tab showing learn session `sessionId`, `setWorkspaceItem` recording what the tab is sent to. */
function renderInLearnTab(element: ReactElement, sessionId: string, setWorkspaceItem = vi.fn()) {
  const tab = { ...nextTab(), appMode: "learn" as const, workspaceItem: { kind: "learn" as const, path: "", projectSlug: null, learn: { kind: "session" as const, sessionId } } }
  return testingRender(
    <TabContext.Provider value={{ tab, isActive: true, setModeLabel: vi.fn(), updateConfig: vi.fn(), setAppMode: vi.fn(), setWorkspaceItem, setHistoryFile: vi.fn(), setTitle: vi.fn() }}>
      {element}
    </TabContext.Provider>,
  )
}

function ompSession(id: string, cwd: string, extra: Partial<import("@/lib/domain/types").OmpSession> = {}) {
  return {
    id, path: `/s/${id}.jsonl`, cwd, name: id, created: "2026-01-01", modified: "2026-01-01", message_count: 1,
    first_message: "", parent_session_id: null, project_root: null, worktree_branch: null, ...extra,
  }
}

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

/** The contents of project "Proj" once the test expanded it. */
function openedProject(view: RenderResult) {
  fireEvent.click(view.getByRole("button", { name: "Expand Proj" }))
  return within(view.getByLabelText("Proj contents"))
}

function WorkspaceLabel() {
  const { tab } = useTab()
  return <output>{tab.name ?? "Initial workspace"}</output>
}

function WorkspaceDraft() {
  const { tab } = useTab()
  return <div data-workspace-surface><input aria-label={`Draft ${tab.name ?? "Initial"}`} defaultValue="" /></div>
}

describe("Project navigation", () => {
  it("renames in place with Enter, blocks duplicate submits, retains failures for retry and cancels with Escape", async () => {
    let resolveRename!: (slug: string) => void
    fixtures.renameProject.mockImplementationOnce(() => new Promise((resolve) => { resolveRename = resolve }))
    const onProjectRenamed = vi.fn()
    const view = render(<Sidebar appMode="chat" onProjectRenamed={onProjectRenamed} />)
    fireEvent.doubleClick(view.getByRole("button", { name: "Proj" }))
    const input = view.getByRole("textbox", { name: "Rename Proj" })
    expect(input).toHaveValue("Proj")
    expect(input).toHaveFocus()
    fireEvent.change(input, { target: { value: "  New project  " } })
    fireEvent.keyDown(input, { key: "Enter" })
    expect(fixtures.renameProject).toHaveBeenCalledWith("proj", "New project")
    expect(input).toHaveAttribute("readonly")
    fireEvent.keyDown(input, { key: "Enter" })
    expect(fixtures.renameProject).toHaveBeenCalledTimes(1)
    await act(async () => resolveRename("new-project"))
    expect(onProjectRenamed).toHaveBeenCalledWith("proj", "new-project")
    expect(view.queryByRole("textbox", { name: "Rename Proj" })).not.toBeInTheDocument()

    fixtures.renameProject.mockRejectedValueOnce(new Error("Name could not be saved"))
    fireEvent.doubleClick(view.getByRole("button", { name: "Proj" }))
    const retry = view.getByRole("textbox", { name: "Rename Proj" })
    fireEvent.change(retry, { target: { value: "Try again" } })
    fireEvent.keyDown(retry, { key: "Enter" })
    expect(await view.findByRole("alert")).toHaveTextContent("Name could not be saved")
    expect(retry).toHaveValue("Try again")
    expect(retry).not.toHaveAttribute("readonly")
    expect(retry).toHaveFocus()
    fireEvent.keyDown(document, { key: "Escape" })
    expect(view.queryByRole("textbox", { name: "Rename Proj" })).not.toBeInTheDocument()
    expect(view.queryByRole("alert")).not.toBeInTheDocument()
    expect(view.getByRole("button", { name: "Proj" })).toBeInTheDocument()
  })

  it("relocates the renamed project's tabs without replacing their mounted drafts", async () => {
    const manager = createRef<TabManagerHandle>()
    const view = testingRender(<TabManager ref={manager}><WorkspaceDraft /></TabManager>)
    const tab = { ...nextTab(), projectSlug: "proj", historyFile: "proj/chat.json", workspaceItem: { kind: "canvas" as const, path: "project/proj/document/board.canvas", projectSlug: "proj" }, appMode: "canvas" as const }
    const other = { ...nextTab(), projectSlug: "proj2", historyFile: "proj2/chat.json" }
    act(() => { manager.current!.openTab(other); manager.current!.openTab(tab) })
    const draft = view.getAllByRole("textbox", { hidden: true }).at(-1)!
    fireEvent.change(draft, { target: { value: "unsaved content" } })
    act(() => manager.current!.relocateProject("proj", "renamed"))
    const tabs = manager.current!.getTabs()
    expect(tabs.find((current) => current.id === tab.id)).toMatchObject({
      id: tab.id, chatId: tab.chatId, historyFile: "renamed/chat.json", projectSlug: "renamed",
      workspaceItem: { path: "project/renamed/document/board.canvas", projectSlug: "renamed" },
    })
    expect(tabs.find((current) => current.id === other.id)).toBe(other)
    expect(draft).toHaveValue("unsaved content")
  })

  it("opens a file on click and renames it on double-click, then follows it in tabs and lists", async () => {
    fixtures.renameDocument.mockResolvedValue({ path: "project/proj/document/ideas.md", name: "ideas.md", mime: "text/markdown", changed: [] })
    const onDocumentSelect = vi.fn()
    const onDocumentRenamed = vi.fn()
    const onDocumentsChanged = vi.fn()
    const view = render(<Sidebar appMode="chat" onDocumentSelect={onDocumentSelect} onDocumentRenamed={onDocumentRenamed} onDocumentsChanged={onDocumentsChanged} />)
    const file = await openedProject(view).findByRole("button", { name: "notes.md" })
    fireEvent.click(file)
    expect(onDocumentSelect).toHaveBeenCalledWith("project/proj/document/notes.md", "proj")
    expect(view.queryByRole("textbox", { name: "Rename notes.md" })).not.toBeInTheDocument()
    expect(view.getByRole("button", { name: "Delete notes.md" })).toBeInTheDocument()

    fireEvent.doubleClick(file)
    const input = view.getByRole("textbox", { name: "Rename notes.md" }) as HTMLInputElement
    expect(input).toHaveValue("notes.md")
    expect([input.selectionStart, input.selectionEnd]).toEqual([0, "notes".length])
    fireEvent.change(input, { target: { value: "ideas" } })
    fireEvent.keyDown(input, { key: "Enter" })

    await waitFor(() => expect(onDocumentRenamed).toHaveBeenCalledWith("project/proj/document/notes.md", "project/proj/document/ideas.md"))
    expect(fixtures.renameDocument).toHaveBeenCalledWith("proj", "project/proj/document/notes.md", "ideas")
    await waitFor(() => expect(onDocumentsChanged).toHaveBeenCalled())
    expect(api.listProjectDocuments.mock.calls.length).toBeGreaterThan(1)
    expect(api.fileRootTree.mock.calls.length).toBeGreaterThan(1)
    expect(view.queryByRole("textbox", { name: "Rename notes.md" })).not.toBeInTheDocument()
  })

  it("announces the files a rename rewrote to follow it", async () => {
    fixtures.renameDocument.mockResolvedValue({ path: "project/proj/document/ideas.md", name: "ideas.md", mime: "text/markdown", changed: ["/work/paper.tex", "/work/intro.tex"] })
    const changed: string[] = []
    const stop = onFileChange(() => true, (path) => changed.push(path))
    const view = render(<Sidebar appMode="chat" onDocumentRenamed={vi.fn()} />)
    fireEvent.doubleClick(await openedProject(view).findByRole("button", { name: "notes.md" }))
    fireEvent.change(view.getByRole("textbox", { name: "Rename notes.md" }), { target: { value: "ideas" } })
    fireEvent.keyDown(view.getByRole("textbox", { name: "Rename notes.md" }), { key: "Enter" })
    await waitFor(() => expect(changed).toEqual(["/work/paper.tex", "/work/intro.tex"]))
    stop()
  })


  it("keeps the input open with the reason when the name is taken, and leaving or Escape keeps the old name", async () => {
    fixtures.renameDocument.mockRejectedValueOnce(new Error("A document with that name already exists there"))
    const onDocumentRenamed = vi.fn()
    const view = render(<Sidebar appMode="chat" onDocumentRenamed={onDocumentRenamed} />)
    fireEvent.doubleClick(await openedProject(view).findByRole("button", { name: "notes.md" }))
    const input = view.getByRole("textbox", { name: "Rename notes.md" })
    fireEvent.change(input, { target: { value: "board.md" } })
    fireEvent.keyDown(input, { key: "Enter" })

    expect(await view.findByRole("alert")).toHaveTextContent("A document with that name already exists there")
    expect(input).toHaveValue("board.md")
    // Leaving with the refused name drops it instead of asking again
    fireEvent.blur(input)
    expect(view.queryByRole("textbox", { name: "Rename notes.md" })).not.toBeInTheDocument()
    expect(fixtures.renameDocument).toHaveBeenCalledTimes(1)

    fireEvent.doubleClick(view.getByRole("button", { name: "notes.md" }))
    fireEvent.change(view.getByRole("textbox", { name: "Rename notes.md" }), { target: { value: "other.md" } })
    fireEvent.keyDown(document, { key: "Escape" })
    expect(view.queryByRole("textbox", { name: "Rename notes.md" })).not.toBeInTheDocument()
    expect(fixtures.renameDocument).toHaveBeenCalledTimes(1)
    expect(onDocumentRenamed).not.toHaveBeenCalled()
  })

  it("renames a mounted file of the Code section outside any project when leaving the input", async () => {
    fixtures.renameDocument.mockResolvedValue({ path: "/source/renamed.md", name: "renamed.md", mime: "text/markdown", changed: [] })
    const onDocumentRenamed = vi.fn()
    const view = render(<Sidebar appMode="chat" onDocumentRenamed={onDocumentRenamed} />)
    const code = within(view.getByLabelText("Code"))
    fireEvent.click(await code.findByRole("button", { name: "source" }))
    fireEvent.doubleClick(code.getByRole("button", { name: "first.md" }))
    const input = code.getByRole("textbox", { name: "Rename first.md" })
    fireEvent.change(input, { target: { value: "renamed.md" } })
    fireEvent.blur(input)

    await waitFor(() => expect(onDocumentRenamed).toHaveBeenCalledWith("/source/first.md", "/source/renamed.md"))
    expect(fixtures.renameDocument).toHaveBeenCalledWith("", "/source/first.md", "renamed.md")
  })

  it("offers no rename on folders and library PDFs", async () => {
    fixtures.documents.push({ path: "PDFs/paper.pdf", name: "paper.pdf", mime: "application/pdf" })
    try {
      const view = render(<Sidebar appMode="chat" />)
      fireEvent.doubleClick(await openedProject(view).findByRole("button", { name: "paper.pdf" }))
      const code = within(view.getByLabelText("Code"))
      fireEvent.click(await code.findByRole("button", { name: "source" }))
      fireEvent.doubleClick(code.getByRole("button", { name: "docs" }))
      expect(view.queryByRole("textbox", { name: /^Rename / })).not.toBeInTheDocument()
    } finally {
      fixtures.documents.pop()
    }
  })

  it("moves a tab showing a renamed file and keeps its workbench session", () => {
    const manager = createRef<TabManagerHandle>()
    testingRender(<TabManager ref={manager}><WorkspaceDraft /></TabManager>)
    const workbench = { ...nextTab(), appMode: "workbench" as const, workspaceItem: { kind: "workbench" as const, path: "/data/workbench/latex/proj/paper/paper.tex", projectSlug: "proj" } }
    const document = { ...nextTab(), appMode: "document" as const, workspaceItem: { kind: "document" as const, path: "project/proj/document/notes.md", projectSlug: "proj" } }
    const other = { ...nextTab(), appMode: "document" as const, workspaceItem: { kind: "document" as const, path: "project/proj/document/other.md", projectSlug: "proj" } }
    act(() => { manager.current!.openTab(workbench); manager.current!.openTab(document); manager.current!.openTab(other) })

    act(() => manager.current!.relocateDocument("/data/workbench/latex/proj/paper/paper.tex", "/data/workbench/latex/proj/paper/thesis.tex"))
    act(() => manager.current!.relocateDocument("project/proj/document/notes.md", "project/proj/document/ideas.md"))

    const tabs = manager.current!.getTabs()
    expect(tabs.find((tab) => tab.id === workbench.id)?.workspaceItem).toEqual({
      kind: "workbench", path: "/data/workbench/latex/proj/paper/thesis.tex", projectSlug: "proj", sessionKey: "/data/workbench/latex/proj/paper/paper.tex",
    })
    expect(tabs.find((tab) => tab.id === document.id)?.workspaceItem?.path).toBe("project/proj/document/ideas.md")
    expect(tabs.find((tab) => tab.id === other.id)).toBe(other)
  })

  it("switches an open LaTeX project's tab to another of its files, and opens files outside in the current tab", () => {
    const manager = createRef<TabManagerHandle>()
    render(<TabManager ref={manager}><TabList /></TabManager>)
    localStorage.setItem("sidebar_tabs_expanded", "true")
    const project = {
      ...nextTab(), appMode: "workbench" as const,
      workspaceItem: { kind: "workbench" as const, path: "/repo/paper/main.tex", projectSlug: null, rootFile: true, texFolder: "/repo/paper" },
    }
    act(() => manager.current!.openTab(project))
    act(() => manager.current!.switchToTab(manager.current!.getTabs()[0].id))
    const count = manager.current!.getTabs().length

    act(() => manager.current!.showItem({ kind: "workbench", path: "/repo/paper/refs.bib", projectSlug: null, rootFile: true }))
    expect(manager.current!.getTabs()).toHaveLength(count)
    expect(manager.current!.getTabs().find((tab) => tab.id === project.id)?.workspaceItem?.path).toBe("/repo/paper/refs.bib")

    act(() => manager.current!.switchToTab(manager.current!.getTabs()[0].id))
    act(() => manager.current!.showItem({ kind: "workbench", path: "/repo/elsewhere/other.tex", projectSlug: null, rootFile: true }))
    expect(manager.current!.getTabs()).toHaveLength(count)
    expect(manager.current!.getTabs().find((tab) => tab.id === project.id)?.workspaceItem?.path).toBe("/repo/paper/refs.bib")
    expect(manager.current!.getTabs()[0].workspaceItem?.path).toBe("/repo/elsewhere/other.tex")
  })

  it("opens an image from a LaTeX project tab in a new tab and leaves the workbench in place", () => {
    const manager = createRef<TabManagerHandle>()
    render(<TabManager ref={manager}><TabList /></TabManager>)
    const project = {
      ...nextTab(), appMode: "workbench" as const,
      workspaceItem: { kind: "workbench" as const, path: "/repo/paper/main.tex", projectSlug: null, rootFile: true, texFolder: "/repo/paper" },
    }
    act(() => manager.current!.openTab(project))
    const count = manager.current!.getTabs().length

    act(() => manager.current!.showItem({ kind: "document", path: "/repo/paper/imgs/plot.png", projectSlug: null, rootFile: true }))
    const tabs = manager.current!.getTabs()
    expect(tabs).toHaveLength(count + 1)
    expect(tabs.find((tab) => tab.id === project.id)?.workspaceItem?.path).toBe("/repo/paper/main.tex")
    expect(tabs.at(-1)).toMatchObject({ appMode: "document", workspaceItem: { kind: "document", path: "/repo/paper/imgs/plot.png" } })
    expect(tabs.at(-1)?.id).not.toBe(project.id)

    act(() => manager.current!.showItem({ kind: "document", path: "/repo/paper/notes.pdf", projectSlug: null, rootFile: true }))
    expect(manager.current!.getTabs()).toHaveLength(count + 1)
    expect(manager.current!.getTabs().at(-1)?.workspaceItem?.path).toBe("/repo/paper/notes.pdf")
  })

  it("offers a new file in mounts and promoted folders that directly hold a .tex, and opens what it creates", async () => {
    const folder = "/data/workbench/latex/proj/paper"
    api.listProjectDocuments.mockImplementation(async (slug) => slug === "proj"
      ? [{
        path: folder, name: "paper", mime: "inode/directory", children: [
          { path: `${folder}/main.tex`, name: "main.tex", type: "file" },
          { path: `${folder}/chapters`, name: "chapters", type: "folder", children: [{ path: `${folder}/chapters/one.TEX`, name: "one.TEX", type: "file" }] },
          { path: `${folder}/imgs`, name: "imgs", type: "folder", children: [] },
        ],
      }]
      : [])
    api.fileRootTree.mockResolvedValue({ enabled: true, tree: [
      { path: "/paper", name: "paper", type: "root", children: [{ path: "/paper/main.tex", name: "main.tex", type: "file" }] },
      { path: "/source", name: "source", type: "root", children: [] },
    ] })
    api.createTexProjectFile.mockResolvedValue(`${folder}/chapters/two.tex`)
    const onFileSelect = vi.fn()
    const view = render(<Sidebar appMode="chat" onFileSelect={onFileSelect} />)
    const code = within(view.getByLabelText("Code"))
    expect(await code.findByRole("button", { name: "New file in paper" })).toBeInTheDocument()
    expect(code.queryByRole("button", { name: "New file in source" })).not.toBeInTheDocument()
    const project = openedProject(view)

    fireEvent.click(await project.findByRole("button", { name: "paper" }))
    expect(project.getByRole("button", { name: "New file in paper" })).toBeInTheDocument()
    fireEvent.click(project.getByRole("button", { name: "chapters" }))
    expect(project.queryByRole("button", { name: "New file in imgs" })).not.toBeInTheDocument()
    fireEvent.click(project.getByRole("button", { name: "New file in chapters" }))
    fireEvent.change(project.getByLabelText("File name"), { target: { value: "two" } })
    fireEvent.keyDown(project.getByLabelText("File name"), { key: "Enter" })
    await waitFor(() => expect(onFileSelect).toHaveBeenLastCalledWith(`${folder}/chapters/two.tex`, "proj"))
    expect(api.createTexProjectFile).toHaveBeenCalledWith(`${folder}/chapters`, "two.tex")
  })

  it("offers a LaTeX project on a project's workbench .tex, also a renamed one, and promotes it", async () => {
    api.listProjectDocuments.mockImplementation(async (slug) => slug === "proj"
      ? [
        { path: "/data/workbench/latex/proj/paper/paper.tex", name: "paper.tex", mime: "text/x-tex" },
        { path: "/data/workbench/latex/proj/test2/test.tex", name: "test.tex", mime: "text/x-tex" },
        { path: "/data/workbench/latex/proj/paper/refs.bib", name: "refs.bib", mime: "text/plain" },
        { path: "/data/workbench/markdown/proj/plan/plan.md", name: "plan.md", mime: "text/markdown" },
        { path: "project/proj/document/notes.md", name: "notes.md", mime: "text/markdown" },
      ]
      : [])
    api.promoteToTexProject.mockResolvedValue("/data/workbench/latex/proj/paper")
    const view = render(<Sidebar appMode="chat" />)
    await openedProject(view).findByText("paper.tex")

    const promote = view.getAllByRole("button", { name: "Make LaTeX project" })
    expect(promote).toHaveLength(2)
    fireEvent.click(promote[1])
    await waitFor(() => expect(api.promoteToTexProject).toHaveBeenCalledWith("proj", "/data/workbench/latex/proj/test2/test.tex"))
  })

  it("opens immediate categories by default and lists mounted roots collapsed under Code", async () => {
    localStorage.setItem("expanded_workspace_projects", "[]")
    const view = render(<Sidebar appMode="chat" />)
    const general = within(view.getByLabelText("general contents"))
    expect(general.getByRole("button", { name: "canvases" })).toHaveAttribute("aria-expanded", "true")
    expect(general.queryByRole("button", { name: "files" })).not.toBeInTheDocument()
    const fileMounts = within(view.getByLabelText("Code"))
    expect(await fileMounts.findByRole("button", { name: "source" })).toHaveAttribute("aria-expanded", "false")
    expect(fileMounts.queryByRole("button", { name: "first.md" })).not.toBeInTheDocument()

    fireEvent.click(view.getByRole("button", { name: "Expand Proj" }))
    const project = within(view.getByLabelText("Proj contents"))
    for (const category of ["chats", "canvases", "files"]) {
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

  it("reopens a file mount with its contents collapsed, ignoring saved expansion", async () => {
    localStorage.setItem("expanded_workspace_projects", JSON.stringify({ "/source": true }))
    const view = render(<Sidebar appMode="chat" />)
    const fileMounts = within(view.getByLabelText("Code"))
    const source = await fileMounts.findByRole("button", { name: "source" })
    expect(source).toHaveAttribute("aria-expanded", "false")
    fireEvent.click(source)
    fireEvent.click(fileMounts.getByRole("button", { name: "docs" }))
    expect(fileMounts.getByRole("button", { name: "guide.md" })).toBeVisible()
    fireEvent.click(source)
    fireEvent.click(source)
    expect(fileMounts.getByRole("button", { name: "docs" })).toHaveAttribute("aria-expanded", "false")
    expect(fileMounts.queryByRole("button", { name: "guide.md" })).not.toBeInTheDocument()
  })

  it("opens the diff of a file mount from its hover action", async () => {
    const onDiffSelect = vi.fn()
    const view = render(<Sidebar appMode="chat" onDiffSelect={onDiffSelect} />)
    fireEvent.click(await within(view.getByLabelText("Code")).findByRole("button", { name: "Open diff for source" }))
    expect(onDiffSelect).toHaveBeenCalledWith("/source")
  })

  it("switches a clone mount to a worktree, then removes it and keeps the branch unless confirmed", async () => {
    const { ApiError } = await vi.importActual<typeof import("@/lib/backend/api")>("@/lib/backend/api")
    const listWorktrees = api.fetchWorktrees.getMockImplementation()
    api.fetchWorktrees.mockImplementation(async () => ({ is_clone: true, default_base: "dev", default_branch: "dev", branches: ["dev", "feature"], worktrees: [
      { path: "/wt/feature", branch: "feature", is_main: false, changed_files: 2, ahead: 1, behind: 0 },
      { path: "/source", branch: "dev", is_main: true, changed_files: 0, ahead: 0, behind: 0 },
    ] }))
    api.removeWorktree.mockRejectedValueOnce(new ApiError("contains modified or untracked files", 409))
    try {
      const view = render(<Sidebar appMode="chat" />)
      const fileMounts = within(view.getByLabelText("Code"))
      fireEvent.click(await fileMounts.findByRole("button", { name: "source" }))
      fireEvent.click(await fileMounts.findByRole("button", { name: /^Worktrees/ }))
      const mainCheckout = fileMounts.getByRole("button", { name: "dev" })
      const feature = fileMounts.getByRole("button", { name: "feature 2 changed ↑1" })
      expect(mainCheckout.compareDocumentPosition(feature) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
      expect(mainCheckout).toHaveAttribute("aria-current", "page")

      fireEvent.click(fileMounts.getByRole("button", { name: "feature 2 changed ↑1" }))

      expect(fileMounts.getByRole("button", { name: "source" })).toBeVisible()
      expect(fileMounts.getByLabelText("Selected checkout: feature")).toBeVisible()
      expect(feature).toHaveAttribute("aria-current", "page")

      fireEvent.click(mainCheckout)
      expect(mainCheckout).toHaveAttribute("aria-current", "page")
      expect(fileMounts.getByLabelText("Selected checkout: dev")).toBeVisible()
      fireEvent.click(feature)
      await waitFor(() => expect(feature).toHaveAttribute("aria-current", "page"))

      fireEvent.click(fileMounts.getByRole("button", { name: "Delete feature" }))
      fireEvent.click(await fileMounts.findByRole("button", { name: "Yes" }))
      expect(api.removeWorktree).toHaveBeenLastCalledWith("/source", "/wt/feature", true)
      fireEvent.keyDown(await fileMounts.findByRole("group", { name: "Delete branch feature?" }), { key: "Escape" })

      expect(fileMounts.queryByRole("group", { name: "Delete branch feature?" })).not.toBeInTheDocument()
      expect(api.deleteWorktreeBranch).not.toHaveBeenCalled()
    } finally {
      api.fetchWorktrees.mockImplementation(listWorktrees!)
    }
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

  it("lists a renamed project file under its display name while it keeps its key", async () => {
    api.listProjectDocuments.mockImplementation(async (slug) => slug === "proj"
      ? [{ path: "project/proj/document/plan.md", name: "Roadmap.md", mime: "text/markdown" }]
      : [])
    const onDocumentSelect = vi.fn()
    const view = render(<Sidebar appMode="chat" onDocumentSelect={onDocumentSelect} />)

    fireEvent.click(view.getByRole("button", { name: "Expand Proj" }))
    const project = within(view.getByLabelText("Proj contents"))
    fireEvent.click(await project.findByText("Roadmap.md"))
    expect(onDocumentSelect).toHaveBeenLastCalledWith("project/proj/document/plan.md", "proj")
    expect(project.queryByText("plan.md")).not.toBeInTheDocument()
  })

  it("opens Markdown and LaTeX files in the workbench where it is available, the rest in the editor", async () => {
    api.listProjectDocuments.mockImplementation(async (slug) => slug === "proj"
      ? [
        { path: "project/proj/document/notes.md", name: "notes.md", mime: "text/markdown" },
        { path: "/data/workbench/latex/proj/paper/paper.tex", name: "paper.tex", mime: "text/x-tex" },
        { path: "project/proj/document/data.csv", name: "data.csv", mime: "text/csv" },
      ]
      : [])
    const onDocumentSelect = vi.fn()
    const onWorkbenchSelect = vi.fn()
    const view = render(<Sidebar appMode="chat" onDocumentSelect={onDocumentSelect} onWorkbenchSelect={onWorkbenchSelect} />)

    fireEvent.click(view.getByRole("button", { name: "Expand Proj" }))
    const project = within(view.getByLabelText("Proj contents"))
    fireEvent.click(await project.findByText("notes.md"))
    expect(onWorkbenchSelect).toHaveBeenLastCalledWith("project/proj/document/notes.md", "proj")
    fireEvent.click(project.getByText("paper.tex"))
    expect(onWorkbenchSelect).toHaveBeenLastCalledWith("/data/workbench/latex/proj/paper/paper.tex", "proj")
    fireEvent.click(project.getByText("data.csv"))
    expect(onDocumentSelect).toHaveBeenCalledExactlyOnceWith("project/proj/document/data.csv", "proj")
  })

  it("reads the project files again once opening moved a stored document to disk", async () => {
    let moved = false
    api.listProjectDocuments.mockImplementation(async (slug) => slug === "proj"
      ? [{ path: moved ? "/data/workbench/markdown/proj/plan/plan.md" : "project/proj/document/plan.md", name: "plan.md", mime: "text/markdown" }]
      : [])
    const onWorkbenchSelect = vi.fn()
    const view = render(<Sidebar appMode="workbench" activeDocumentPath="project/proj/document/plan.md" onWorkbenchSelect={onWorkbenchSelect} />)
    fireEvent.click(view.getByRole("button", { name: "Expand Proj" }))
    await within(view.getByLabelText("Proj contents")).findByText("plan.md")

    moved = true
    view.rerender(<TabContext.Provider value={{ tab: nextTab(), isActive: true, setModeLabel: vi.fn(), updateConfig: vi.fn(), setAppMode: vi.fn(), setWorkspaceItem: vi.fn(), setHistoryFile: vi.fn(), setTitle: vi.fn() }}>
      <Sidebar appMode="workbench" activeDocumentPath="/data/workbench/markdown/proj/plan/plan.md" onWorkbenchSelect={onWorkbenchSelect} />
    </TabContext.Provider>)
    await waitFor(() => {
      fireEvent.click(within(view.getByLabelText("Proj contents")).getByText("plan.md"))
      expect(onWorkbenchSelect).toHaveBeenLastCalledWith("/data/workbench/markdown/proj/plan/plan.md", "proj")
    })
  })

  it("hides a PDF's extracted Markdown and deletes both together", async () => {
    api.listProjectDocuments.mockImplementation(async (slug) => slug === "proj"
      ? [
        { path: "PDFs/Paper.pdf", name: "Paper.pdf", mime: "application/pdf" },
        { path: "project/proj/document/Paper.md", name: "Paper.md", mime: "text/markdown" },
      ]
      : [])
    vi.mocked(removeDocument).mockResolvedValue([])
    const view = render(<Sidebar appMode="chat" />)
    fireEvent.click(view.getByRole("button", { name: "Expand Proj" }))
    const project = within(view.getByLabelText("Proj contents"))
    await project.findByText("Paper.pdf")
    expect(project.queryByText("Paper.md")).toBeNull()

    expect(project.queryByRole("button", { name: "Expand Paper.pdf" })).toBeNull()

    fireEvent.click(project.getByRole("button", { name: "Delete Paper.pdf" }))
    await undo.schedule.mock.lastCall![2]()
    expect(removeDocument).toHaveBeenCalledWith("proj", "PDFs/Paper.pdf")
    expect(removeDocument).toHaveBeenCalledWith("proj", "project/proj/document/Paper.md")
  })

  it("creates a blank LaTeX file and a LaTeX file from a template on disk and a Markdown file in the project, all opening in the workbench", async () => {
    const onWorkbenchSelect = vi.fn()
    const view = render(<Sidebar appMode="chat" onWorkbenchSelect={onWorkbenchSelect} />)
    fireEvent.click(view.getByRole("button", { name: "Expand Proj" }))
    const project = within(view.getByLabelText("Proj contents"))
    await waitFor(() => expect(api.loadTemplateSets).toHaveBeenCalled())

    fireEvent.click(await project.findByRole("button", { name: "New file in Proj" }))
    const types = within(await view.findByRole("listbox", { name: "File type" }))
    expect(types.getAllByRole("option").map((option) => option.textContent)).toEqual(["Markdown", "LaTeX"])
    expect(types.getByRole("option", { name: "Markdown" })).toHaveAttribute("aria-selected", "true")
    fireEvent.keyDown(view.getByRole("listbox", { name: "File type" }), { key: "Tab" })
    fireEvent.keyDown(view.getByRole("listbox", { name: "File type" }), { key: "Enter" })
    const templates = within(await view.findByRole("listbox", { name: "Template" }))
    expect(templates.getAllByRole("option").map((option) => option.textContent)).toEqual(["Blank", "Deck"])
    expect(templates.getByRole("option", { name: "Blank" })).toHaveAttribute("aria-selected", "true")
    fireEvent.keyDown(view.getByRole("listbox", { name: "Template" }), { key: "Enter" })
    fireEvent.change(project.getByLabelText("File name"), { target: { value: "paper" } })
    fireEvent.keyDown(project.getByLabelText("File name"), { key: "Enter" })
    await waitFor(() => expect(onWorkbenchSelect).toHaveBeenLastCalledWith("/data/workbench/latex/proj/paper/paper.tex", "proj"))
    expect(api.createWorkbenchDocument).toHaveBeenCalledWith("proj", "paper.tex", undefined)

    fireEvent.click(project.getByRole("button", { name: "New file in Proj" }))
    fireEvent.click(within(await view.findByRole("listbox", { name: "File type" })).getByRole("option", { name: "LaTeX" }))
    fireEvent.click(within(await view.findByRole("listbox", { name: "Template" })).getByRole("option", { name: "Deck" }))
    fireEvent.change(project.getByLabelText("File name"), { target: { value: "deck" } })
    fireEvent.keyDown(project.getByLabelText("File name"), { key: "Enter" })
    await waitFor(() => expect(onWorkbenchSelect).toHaveBeenLastCalledWith("/data/workbench/latex/proj/deck/main.tex", "proj"))
    expect(api.createWorkbenchDocument).toHaveBeenLastCalledWith("proj", "deck.tex", "deck-set")

    fireEvent.click(project.getByRole("button", { name: "New file in Proj" }))
    fireEvent.keyDown(await view.findByRole("listbox", { name: "File type" }), { key: "Enter" })
    expect(view.queryByRole("listbox", { name: "Template" })).not.toBeInTheDocument()
    fireEvent.change(project.getByLabelText("File name"), { target: { value: "draft" } })
    fireEvent.keyDown(project.getByLabelText("File name"), { key: "Enter" })
    await waitFor(() => expect(onWorkbenchSelect).toHaveBeenLastCalledWith("project/proj/document/draft.md", "proj"))
    expect(api.writeDocument).toHaveBeenCalledWith("proj", "draft.md")
  })

  it("lists every template set under its label after LaTeX and creates from the picked set", async () => {
    api.loadTemplateSets.mockResolvedValueOnce([{ name: "presentation", label: "Presentation" }, { name: "letter", label: "Letter" }])
    const onWorkbenchSelect = vi.fn()
    const view = render(<Sidebar appMode="chat" onWorkbenchSelect={onWorkbenchSelect} />)
    fireEvent.click(view.getByRole("button", { name: "Expand Proj" }))
    const project = within(view.getByLabelText("Proj contents"))
    await waitFor(() => expect(api.loadTemplateSets).toHaveBeenCalled())

    fireEvent.click(await project.findByRole("button", { name: "New file in Proj" }))
    fireEvent.click(within(await view.findByRole("listbox", { name: "File type" })).getByRole("option", { name: "LaTeX" }))
    const templates = within(await view.findByRole("listbox", { name: "Template" }))
    expect(templates.getAllByRole("option").map((option) => option.textContent)).toEqual(["Blank", "Presentation", "Letter"])
    fireEvent.click(templates.getByRole("option", { name: "Letter" }))
    fireEvent.change(project.getByLabelText("File name"), { target: { value: "cover" } })
    fireEvent.keyDown(project.getByLabelText("File name"), { key: "Enter" })
    await waitFor(() => expect(api.createWorkbenchDocument).toHaveBeenLastCalledWith("proj", "cover.tex", "letter"))
  })

  it("creates a blank LaTeX file without a template choice when the template sets fail to load", async () => {
    api.loadTemplateSets.mockRejectedValueOnce(new Error("offline"))
    const view = render(<Sidebar appMode="chat" onWorkbenchSelect={vi.fn()} />)
    fireEvent.click(view.getByRole("button", { name: "Expand Proj" }))
    const project = within(view.getByLabelText("Proj contents"))
    await waitFor(() => expect(api.loadTemplateSets).toHaveBeenCalled())

    fireEvent.click(await project.findByRole("button", { name: "New file in Proj" }))
    fireEvent.click(within(await view.findByRole("listbox", { name: "File type" })).getByRole("option", { name: "LaTeX" }))
    expect(view.queryByRole("listbox", { name: "Template" })).not.toBeInTheDocument()
    fireEvent.change(project.getByLabelText("File name"), { target: { value: "notes" } })
    fireEvent.keyDown(project.getByLabelText("File name"), { key: "Enter" })
    await waitFor(() => expect(api.createWorkbenchDocument).toHaveBeenLastCalledWith("proj", "notes.tex", undefined))
  })

  it("creates the architecture of a project without one when its icon is used", async () => {
    const onArchitectureSelect = vi.fn()
    const view = render(<Sidebar appMode="chat" onArchitectureSelect={onArchitectureSelect} />)

    fireEvent.click(await view.findByRole("button", { name: "Create architecture for Empty project" }))

    await waitFor(() => expect(onArchitectureSelect).toHaveBeenCalledWith("empty"))
    expect(vi.mocked((await import("@/lib/backend/api")).createArchitectureSource)).toHaveBeenCalledWith("empty", "system.c4")
  })

  it("lists a mount's sessions (worktrees included) under its Sessions, not under the project named like it, and keeps the project link", async () => {
    const listTree = api.fileRootTree.getMockImplementation()
    api.fileRootTree.mockImplementation(async () => ({ enabled: true, tree: [{ path: "/git/proj", name: "proj", type: "root", children: [] }] }))
    omp.overview = {
      running_session_ids: ["fork"],
      workspaces: [],
      sessions: [
        ompSession("main", "/git/proj/src"),
        ompSession("fork", "/git/proj", { parent_session_id: "main" }),
        ompSession("worktree", "/git/proj-worktrees/feat", { project_root: "/git/proj" }),
        ompSession("elsewhere", "/git/project-old"),
      ],
    }
    try {
      const onItemOpen = vi.fn()
      const view = render(<Sidebar appMode="chat" onItemOpen={onItemOpen} />)
      fireEvent.click(view.getByRole("button", { name: "Expand Proj" }))
      expect(within(view.getByLabelText("Proj contents")).queryByRole("button", { name: "Sessions" })).not.toBeInTheDocument()
      const section = within(view.getByLabelText("Code"))
      fireEvent.click(await section.findByRole("button", { name: "proj" }))
      fireEvent.click(within(section.getByRole("group", { name: "proj contents" })).getByRole("button", { name: "Sessions" }))
      const code = within(section.getByRole("group", { name: "Sessions contents" }))
      expect(code.getByRole("button", { name: "worktree" })).toBeVisible()
      expect(code.queryByRole("button", { name: "elsewhere" })).not.toBeInTheDocument()

      fireEvent.click(code.getByRole("button", { name: "main" }))
      expect(onItemOpen).toHaveBeenLastCalledWith({ kind: "omp", path: "/git/proj", projectSlug: "proj", omp: { kind: "session", sessionId: "main" } })
      expect(code.getByRole("button", { name: "fork running" })).toBeVisible()

      fireEvent.click(section.getByRole("button", { name: "New session in proj" }))
      expect(onItemOpen).toHaveBeenLastCalledWith({ kind: "omp", path: "/git/proj", projectSlug: "proj", omp: { kind: "new", cwd: "/git/proj" } }, true)
    } finally {
      omp.overview = null
      api.fileRootTree.mockImplementation(listTree!)
    }
  })

  it("lists omp-only directories beside the mounts, each with its sessions under Sessions, and hides them while omp-web never answered", async () => {
    const listTree = api.fileRootTree.getMockImplementation()
    api.fileRootTree.mockImplementation(async () => ({ enabled: true, tree: [{ path: "/git/proj", name: "proj", type: "root", children: [] }] }))
    omp.overview = {
      running_session_ids: [],
      workspaces: [{ path: "/git/proj", alias: null, added_at: null }, { path: "/git/tool", alias: null, added_at: null }],
      sessions: [ompSession("linked", "/git/proj"), ompSession("tooling", "/git/tool/src")],
    }
    try {
      const onItemOpen = vi.fn()
      const view = render(<Sidebar appMode="chat" onItemOpen={onItemOpen} />)
      const section = within(view.getByLabelText("Code"))
      fireEvent.click(await section.findByRole("button", { name: "proj" }))
      const mount = within(section.getByRole("group", { name: "proj contents" }))
      fireEvent.click(mount.getByRole("button", { name: "Sessions" }))
      expect(mount.getByRole("button", { name: "linked" })).toBeVisible()
      expect(mount.queryByRole("button", { name: "tooling" })).not.toBeInTheDocument()

      fireEvent.click(section.getByRole("button", { name: "tool" }))
      const tool = within(section.getByRole("group", { name: "tool contents" }))
      fireEvent.click(tool.getByRole("button", { name: "Sessions" }))
      expect(tool.getByRole("button", { name: "tooling" })).toBeVisible()
      expect(tool.queryByRole("button", { name: "linked" })).not.toBeInTheDocument()

      fireEvent.click(section.getByRole("button", { name: "Open omp settings" }))
      expect(onItemOpen).toHaveBeenLastCalledWith({ kind: "omp", path: "", projectSlug: null, omp: { kind: "settings" } })
      fireEvent.click(section.getByRole("button", { name: "New session in tool" }))
      expect(onItemOpen).toHaveBeenLastCalledWith({ kind: "omp", path: "/git/tool", projectSlug: null, omp: { kind: "new", cwd: "/git/tool" } }, true)
      fireEvent.click(tool.getByRole("button", { name: "tooling" }))
      expect(onItemOpen).toHaveBeenLastCalledWith({ kind: "omp", path: "/git/tool", projectSlug: null, omp: { kind: "session", sessionId: "tooling" } })

      omp.overview = null
      view.rerender(<TabContext.Provider value={{ tab: nextTab(), isActive: true, setModeLabel: vi.fn(), updateConfig: vi.fn(), setAppMode: vi.fn(), setWorkspaceItem: vi.fn(), setHistoryFile: vi.fn(), setTitle: vi.fn() }}>
        <Sidebar appMode="chat" onItemOpen={onItemOpen} />
      </TabContext.Provider>)
      expect(section.queryByRole("button", { name: "tool" })).not.toBeInTheDocument()
      expect(section.queryByRole("button", { name: "Open omp settings" })).not.toBeInTheDocument()
      expect(section.getByRole("button", { name: "proj" })).toBeVisible()
    } finally {
      omp.overview = null
      api.fileRootTree.mockImplementation(listTree!)
    }
  })

  it("archives a directory like a project, listing it in the header popover until restored, for mounts and omp-only directories alike", async () => {
    omp.overview = {
      running_session_ids: [],
      workspaces: [{ path: "/git/tool", alias: null, added_at: null }],
      sessions: [],
    }
    try {
      const view = render(<Sidebar appMode="chat" />)
      const section = within(view.getByLabelText("Code"))
      await section.findByRole("button", { name: "source" })

      fireEvent.click(section.getByRole("button", { name: "Archive source" }))
      fireEvent.click(section.getByRole("button", { name: "Archive tool" }))
      expect(section.queryByRole("button", { name: "source" })).not.toBeInTheDocument()
      expect(section.queryByRole("button", { name: "tool" })).not.toBeInTheDocument()

      fireEvent.click(section.getByRole("button", { name: "Show all directories" }))
      fireEvent.click(await view.findByText("source"))
      expect(await section.findByRole("button", { name: "source" })).toBeVisible()
      expect(section.queryByRole("button", { name: "tool" })).not.toBeInTheDocument()
    } finally {
      omp.overview = null
    }
  })

  it("opens the factory of any workspace, registering its checkout first unless the engine knows it, for host access only", async () => {
    omp.overview = {
      running_session_ids: [],
      workspaces: [{ path: "/git/known", alias: null, added_at: null }, { path: "/git/fresh", alias: null, added_at: null }, { path: "/git/broken", alias: null, added_at: null }],
      sessions: [],
    }
    login.hostAccess = true
    api.fetchFactoryRepos.mockResolvedValue([{ name: "known-repo", path: "/git/known" }])
    api.registerFactoryRepo.mockResolvedValueOnce({ name: "fresh", path: "/git/fresh" }).mockRejectedValueOnce(new Error("Not a git checkout: /git/broken"))
    try {
      const onItemOpen = vi.fn()
      const view = render(<Sidebar appMode="chat" onItemOpen={onItemOpen} />)
      await waitFor(() => expect(api.fetchFactoryRepos).toHaveBeenCalled())
      await act(async () => {})

      fireEvent.click(view.getByRole("button", { name: "Open factory for known" }))
      await waitFor(() => expect(onItemOpen).toHaveBeenLastCalledWith({ kind: "factory", path: "known-repo", projectSlug: null }))
      expect(api.registerFactoryRepo).not.toHaveBeenCalled()

      fireEvent.click(view.getByRole("button", { name: "Open factory for fresh" }))
      await waitFor(() => expect(onItemOpen).toHaveBeenLastCalledWith({ kind: "factory", path: "fresh", projectSlug: null }))
      expect(api.registerFactoryRepo).toHaveBeenCalledWith("/git/fresh")

      fireEvent.click(view.getByRole("button", { name: "Open factory for fresh" }))
      await waitFor(() => expect(onItemOpen).toHaveBeenCalledTimes(3))
      expect(api.registerFactoryRepo).toHaveBeenCalledTimes(1)

      fireEvent.click(view.getByRole("button", { name: "Open factory for broken" }))
      expect(await view.findByRole("alert")).toHaveTextContent("Not a git checkout: /git/broken")
      expect(onItemOpen).toHaveBeenCalledTimes(3)

      login.hostAccess = false
      view.rerender(<TabContext.Provider value={{ tab: nextTab(), isActive: true, setModeLabel: vi.fn(), updateConfig: vi.fn(), setAppMode: vi.fn(), setWorkspaceItem: vi.fn(), setHistoryFile: vi.fn(), setTitle: vi.fn() }}>
        <Sidebar appMode="chat" onItemOpen={onItemOpen} />
      </TabContext.Provider>)
      expect(view.queryByRole("button", { name: /Open factory for/ })).not.toBeInTheDocument()
    } finally {
      omp.overview = null
      login.hostAccess = false
    }
  })

  it("offers Learn to host logins only: a new session, the knowledge graph and practice", async () => {
    const onItemOpen = vi.fn()
    const guest = render(<Sidebar appMode="chat" onItemOpen={onItemOpen} />)
    await act(async () => {})
    expect(guest.queryByRole("button", { name: "Open learn" })).not.toBeInTheDocument()
    expect(guest.queryByRole("button", { name: "New teaching session" })).not.toBeInTheDocument()
    guest.unmount()

    login.hostAccess = true
    try {
      const view = render(<Sidebar appMode="chat" onItemOpen={onItemOpen} />)
      await act(async () => {})
      fireEvent.click(view.getByRole("button", { name: "Open learn" }))
      expect(onItemOpen).toHaveBeenLastCalledWith({ kind: "learn", path: "", projectSlug: null, learn: { kind: "new" } })
      fireEvent.click(view.getByRole("button", { name: "Open knowledge graph" }))
      expect(onItemOpen).toHaveBeenLastCalledWith({ kind: "learn", path: "", projectSlug: null, learn: { kind: "home" } })
      fireEvent.click(view.getByRole("button", { name: "New teaching session" }))
      expect(onItemOpen).toHaveBeenLastCalledWith({ kind: "learn", path: "", projectSlug: null, learn: { kind: "new" } }, true)
      fireEvent.click(view.getByRole("button", { name: "New practice session" }))
      expect(onItemOpen).toHaveBeenLastCalledWith({ kind: "learn", path: "", projectSlug: null, learn: { kind: "new", sessionKind: "practice" } }, true)

      // A project without learn sessions shows no learn folder, only the action starting one.
      fireEvent.click(view.getByRole("button", { name: "Expand Proj" }))
      expect(within(view.getByLabelText("Proj contents")).queryByRole("button", { name: "learn" })).not.toBeInTheDocument()
      expect(view.getByRole("button", { name: "New learn session in Proj" })).toBeInTheDocument()
    } finally {
      login.hostAccess = false
    }
  })

  it("groups learn sessions under their project, lists the rest under Learn, and opens the frame at a clicked session", async () => {
    learn.sessions.sessions = [learnSession("chat_a", "Fourier series", { project_slug: "proj" }), learnSession("chat_b", "Eigenvalues")]
    const onItemOpen = vi.fn()
    const guest = render(<Sidebar appMode="chat" onItemOpen={onItemOpen} />)
    expect(guest.queryByRole("button", { name: "New learn session in Proj" })).not.toBeInTheDocument()
    expect(guest.queryByRole("button", { name: "Eigenvalues" })).not.toBeInTheDocument()
    guest.unmount()

    login.hostAccess = true
    try {
      const view = render(<Sidebar appMode="chat" onItemOpen={onItemOpen} />)
      await act(async () => {})
      fireEvent.click(view.getByRole("button", { name: "Expand Proj" }))
      const project = within(view.getByLabelText("Proj contents"))
      expect(project.getByRole("button", { name: "learn" })).toHaveAttribute("aria-expanded", "true")
      fireEvent.click(project.getByRole("button", { name: "Fourier series" }))
      expect(onItemOpen).toHaveBeenLastCalledWith({ kind: "learn", path: "", projectSlug: "proj", learn: { kind: "session", sessionId: "chat_a" } })
      fireEvent.click(view.getByRole("button", { name: "New learn session in Proj" }))
      expect(onItemOpen).toHaveBeenLastCalledWith({ kind: "learn", path: "", projectSlug: "proj", learn: { kind: "new" } }, true)

      const section = within(view.getByLabelText("Learn"))
      expect(section.queryByRole("button", { name: "Fourier series" })).not.toBeInTheDocument()
      fireEvent.click(section.getByRole("button", { name: "Eigenvalues" }))
      expect(onItemOpen).toHaveBeenLastCalledWith({ kind: "learn", path: "", projectSlug: null, learn: { kind: "session", sessionId: "chat_b" } })
    } finally {
      login.hostAccess = false
      learn.sessions.sessions = []
    }
  })

  it("opens LearnHarness's settings in the frame, keeping a frame at another version on its own settings", async () => {
    login.hostAccess = true
    try {
      const view = render(<Sidebar appMode="chat" onItemOpen={vi.fn()} />)
      await act(async () => {})
      fireEvent.click(view.getByRole("button", { name: "Open learn settings" }))
      expect(learn.showInFrame.mock.calls).toEqual([[{ type: "open-settings", section: null }]])
      view.unmount()

      learnFrame.drive = "mismatch"
      const mismatched = render(<Sidebar appMode="chat" onItemOpen={vi.fn()} />)
      expect(mismatched.queryByRole("button", { name: "Open learn settings" })).not.toBeInTheDocument()
    } finally {
      login.hostAccess = false
      learnFrame.drive = "matched"
      learn.showInFrame.mockClear()
    }
  })

  it("renames and deletes learn sessions, showing the hidden chrome's state, practice tag or phase chip", async () => {
    const fiveMinutesAgo = new Date(Date.now() - 5 * 60_000).toISOString()
    learn.sessions.sessions = [
      learnSession("chat_a", "Fourier series", { project_slug: "proj" }),
      learnSession("chat_b", "Eigenvalues", { kind: "practice" }),
      learnSession("chat_c", "Pendulum"),
      learnSession("chat_d", "Waves", { pending_question: true, busy: true, updated_at: fiveMinutesAgo }),
      learnSession("chat_e", "Optics", { busy: true }),
    ]
    learnFrame.metas = {
      chat_c: sessionMeta("chat_c", "Pendulum", { phase: "plan" }),
      chat_e: sessionMeta("chat_e", "Optics", { status: "compacting", updatedAt: fiveMinutesAgo }),
    }
    login.hostAccess = true
    try {
      const view = render(<Sidebar appMode="chat" onItemOpen={vi.fn()} />)
      await act(async () => {})
      const section = within(view.getByLabelText("Learn"))
      expect(section.getByRole("button", { name: "Eigenvalues practice" })).toBeInTheDocument()
      expect(section.getByRole("button", { name: "Pendulum plan" })).toBeInTheDocument()
      expect(section.getByRole("button", { name: "Waves waiting 5m" })).toBeInTheDocument()
      expect(section.getByRole("button", { name: "Optics compacting 5m" })).toBeInTheDocument()

      fireEvent.doubleClick(section.getByRole("button", { name: "Eigenvalues practice" }))
      const input = section.getByRole("textbox", { name: "Rename Eigenvalues" })
      expect(input).toHaveValue("Eigenvalues")
      fireEvent.change(input, { target: { value: "Linear algebra" } })
      fireEvent.keyDown(input, { key: "Enter" })
      await waitFor(() => expect(section.queryByRole("textbox", { name: "Rename Eigenvalues" })).not.toBeInTheDocument())
      expect(learn.sessions.rename).toHaveBeenCalledWith("chat_b", "Linear algebra")

      fireEvent.click(section.getByRole("button", { name: "Delete Pendulum" }))
      const [key, label, run] = undo.schedule.mock.lastCall!
      expect([key, label]).toEqual(["learn:chat_c", "Pendulum"])
      await run()
      expect(learn.sessions.remove).toHaveBeenCalledWith("chat_c")
    } finally {
      login.hostAccess = false
      learn.sessions.sessions = []
      learnFrame.metas = {}
    }
  })

  it("filters learn sessions by LearnHarness's ranking and kind, Enter opens the walked match", async () => {
    learn.sessions.sessions = [
      learnSession("chat_a", "Series and Fourier", { project_slug: "proj" }),
      learnSession("chat_b", "Fourier practice", { kind: "practice" }),
      learnSession("chat_c", "Fourier series"),
      learnSession("chat_d", "Pendulum"),
    ]
    const onItemOpen = vi.fn()
    login.hostAccess = true
    try {
      const view = render(<Sidebar appMode="chat" onItemOpen={onItemOpen} />)
      await act(async () => {})
      const section = within(view.getByLabelText("Learn"))
      fireEvent.click(view.getByRole("button", { name: "Filter learn sessions" }))
      const search = view.getByRole("searchbox", { name: "Filter learn sessions" })
      fireEvent.change(search, { target: { value: "fourier ser" } })
      // Every token must match, a title starting with the query ranks first, a project's session names its project.
      const rows = () => section.getAllByRole("button", { name: /^(Proj )?(Fourier|Series)/ }).map((row) => row.textContent)
      expect(rows()).toEqual(["Fourier series", "Proj Series and Fourier"])
      expect(section.getByRole("button", { name: "Fourier series" })).toHaveAttribute("aria-current", "page")

      fireEvent.keyDown(search, { key: "ArrowDown" })
      fireEvent.keyDown(search, { key: "Enter" })
      expect(onItemOpen).toHaveBeenLastCalledWith({ kind: "learn", path: "", projectSlug: "proj", learn: { kind: "session", sessionId: "chat_a" } })

      fireEvent.change(search, { target: { value: "fourier" } })
      fireEvent.click(view.getByRole("button", { name: "Practice" }))
      expect(rows()).toEqual(["Fourier practicepractice"])
      expect(localStorage.getItem("learn_session_kind_filter")).toBe("practice")
      fireEvent.click(view.getByRole("button", { name: "All" }))
      expect(rows()).toHaveLength(3)

      fireEvent.keyDown(document, { key: "Escape" })
      expect(view.queryByRole("searchbox", { name: "Filter learn sessions" })).not.toBeInTheDocument()
      expect(section.queryByRole("button", { name: "Proj Series and Fourier" })).not.toBeInTheDocument()
    } finally {
      login.hostAccess = false
      learn.sessions.sessions = []
    }
  })

  it("opens upstream's session panels in the frame and exports or deletes with the graph traces from a session's menu", async () => {
    learn.sessions.sessions = [learnSession("chat_a", "Fourier series", { project_slug: "proj" }), learnSession("chat_b", "Eigenvalues")]
    learnFrame.plans = { chat_b: { type: "session-plan", sessionId: "chat_b", available: true } }
    api.exportLearnSession.mockResolvedValue({ filename: "eigenvalues.md", markdown: "# Eigenvalues" })
    const createObjectURL = vi.fn(() => "blob:export")
    vi.stubGlobal("URL", Object.assign(URL, { createObjectURL, revokeObjectURL: vi.fn() }))
    const clicked = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {})
    login.hostAccess = true
    try {
      const view = render(<Sidebar appMode="chat" onItemOpen={vi.fn()} />)
      await act(async () => {})
      const section = within(view.getByLabelText("Learn"))
      const menu = () => {
        fireEvent.click(section.getByRole("button", { name: "More actions for Eigenvalues" }))
        return within(view.getByRole("menu", { name: "Session actions" }))
      }
      expect(menu().getAllByRole("menuitem").map((item) => item.textContent)).toEqual(["Plan", "Coverage", "Export as Markdown", "Delete with graph traces"])
      fireEvent.keyDown(document, { key: "Escape" })
      expect(view.queryByRole("menu", { name: "Session actions" })).not.toBeInTheDocument()

      const eigenvalues = learn.sessions.sessions[1]
      for (const panel of ["Plan", "Coverage"]) fireEvent.click(menu().getByRole("menuitem", { name: panel }))
      expect(learn.showInFrame.mock.calls).toEqual(["plan", "coverage"].map((panel) => [{ type: "open-panel", panel, sessionId: "chat_b" }, eigenvalues]))

      fireEvent.click(menu().getByRole("menuitem", { name: "Export as Markdown" }))
      await waitFor(() => expect(clicked).toHaveBeenCalledOnce())
      expect(api.exportLearnSession).toHaveBeenCalledWith("chat_b")
      expect(clicked.mock.contexts[0]).toMatchObject({ download: "eigenvalues.md", href: "blob:export" })

      fireEvent.click(menu().getByRole("menuitem", { name: "Delete with graph traces" }))
      const [key, label, run] = undo.schedule.mock.lastCall!
      expect([key, label]).toEqual(["learn:chat_b", "Eigenvalues and its graph traces"])
      await run()
      expect(learn.sessions.remove).toHaveBeenLastCalledWith("chat_b", "delete-nodes")

      // No plan reported: no Plan entry.
      fireEvent.click(view.getByRole("button", { name: "Expand Proj" }))
      fireEvent.click(view.getByRole("button", { name: "More actions for Fourier series" }))
      expect(within(view.getByRole("menu", { name: "Session actions" })).queryByRole("menuitem", { name: "Plan" })).not.toBeInTheDocument()
    } finally {
      login.hostAccess = false
      learn.sessions.sessions = []
      learnFrame.plans = {}
      learn.showInFrame.mockClear()
      clicked.mockRestore()
      vi.unstubAllGlobals()
    }
  })

  it("lists the frame's background agents and opens upstream's section with their details and stop buttons", async () => {
    learn.sessions.sessions = [learnSession("chat_b", "Eigenvalues")]
    learnFrame.activity = { type: "activity", items: [{ id: "job_1", label: "Extracting Eigenvalues", status: "running", sessionId: "chat_b" }] }
    const onItemOpen = vi.fn()
    login.hostAccess = true
    try {
      const view = render(<Sidebar appMode="chat" onItemOpen={onItemOpen} />)
      await act(async () => {})
      const section = within(view.getByLabelText("Learn"))
      fireEvent.click(section.getByRole("button", { name: "background agents" }))
      fireEvent.click(section.getByRole("button", { name: "Extracting Eigenvalues running" }))
      expect(onItemOpen).toHaveBeenLastCalledWith({ kind: "learn", path: "", projectSlug: null, learn: { kind: "session", sessionId: "chat_b" } })
      fireEvent.click(section.getByRole("button", { name: "Show background agents" }))
      expect(learn.showInFrame).toHaveBeenLastCalledWith({ type: "open-panel", panel: "activity", sessionId: null })
    } finally {
      login.hostAccess = false
      learn.sessions.sessions = []
      learnFrame.activity = null
      learn.showInFrame.mockClear()
    }
  })

  it("acts on the hidden chrome's shortcuts from the frame and on F2, and leaves a session deleted here", async () => {
    learn.sessions.sessions = [learnSession("chat_b", "Eigenvalues")]
    login.hostAccess = true
    try {
      const setWorkspaceItem = vi.fn()
      const view = renderInLearnTab(<Sidebar appMode="learn" onItemOpen={vi.fn()} />, "chat_b", setWorkspaceItem)
      await act(async () => {})
      const press = (action: FrameMessageOf<"chrome-key">["action"]) => act(() => learnFrame.chromeKeys.forEach((listener) => listener({ type: "chrome-key", action })))

      await press("toggle-sidebar")
      expect(sidebarFrame.toggleCollapsed).toHaveBeenCalledOnce()
      await press("sessions-search")
      expect(view.getByRole("searchbox", { name: "Filter learn sessions" })).toHaveFocus()
      await press("rename-session")
      expect(view.getByRole("textbox", { name: "Rename Eigenvalues" })).toHaveValue("Eigenvalues")
      fireEvent.keyDown(document, { key: "Escape" })
      expect(view.queryByRole("textbox", { name: "Rename Eigenvalues" })).not.toBeInTheDocument()
      expect(view.getByRole("searchbox", { name: "Filter learn sessions" })).toBeInTheDocument()

      fireEvent.keyDown(window, { key: "F2" })
      expect(view.getByRole("textbox", { name: "Rename Eigenvalues" })).toHaveValue("Eigenvalues")
      expect(setWorkspaceItem).not.toHaveBeenCalled()
      view.unmount()

      learn.sessions.missing = new Set(["chat_b"])
      renderInLearnTab(<Sidebar appMode="learn" onItemOpen={vi.fn()} />, "chat_b", setWorkspaceItem)
      await act(async () => {})
      expect(setWorkspaceItem).toHaveBeenCalledWith({ kind: "learn", path: "", projectSlug: null, learn: { kind: "home" } })
    } finally {
      login.hostAccess = false
      learn.sessions.sessions = []
      learn.sessions.missing = new Set()
      sidebarFrame.toggleCollapsed.mockClear()
    }
  })

  it("archives a session from its row, explains a refusal for sessions with forks and restores from the archive", async () => {
    const { ApiError } = await vi.importActual<typeof import("@/lib/backend/api")>("@/lib/backend/api")
    omp.overview = {
      running_session_ids: [],
      workspaces: [{ path: "/git/tool", alias: null, added_at: null }],
      sessions: [ompSession("parent", "/git/tool"), ompSession("child", "/git/tool", { parent_session_id: "parent" }), ompSession("solo", "/git/tool")],
    }
    api.fetchOmpArchive.mockImplementation(async () => [{ key: "old.jsonl", id: "old", cwd: "/git/tool", name: "old", created: "", archived_at: "", message_count: 1, first_message: "", status: null }])
    api.archiveOmpSession.mockRejectedValueOnce(new ApiError("session_has_children", 409))
    try {
      const view = render(<Sidebar appMode="chat" />)
      const section = within(view.getByLabelText("Code"))
      fireEvent.click(section.getByRole("button", { name: "tool" }))
      fireEvent.click(within(section.getByRole("group", { name: "tool contents" })).getByRole("button", { name: "Sessions" }))

      fireEvent.click(section.getByRole("button", { name: "Archive parent" }))
      expect(await view.findByRole("alert")).toHaveTextContent("Archive the forks of parent first")
      expect(section.getByRole("button", { name: "parent" })).toBeVisible()

      fireEvent.click(section.getByRole("button", { name: "Archive solo" }))
      await waitFor(() => expect(section.queryByRole("button", { name: "solo" })).not.toBeInTheDocument())
      expect(api.archiveOmpSession).toHaveBeenLastCalledWith("solo")
      expect(view.queryByRole("alert")).not.toBeInTheDocument()

      fireEvent.click(await section.findByRole("button", { name: "archived" }))
      fireEvent.click(section.getByRole("button", { name: "Restore old" }))
      await waitFor(() => expect(api.restoreOmpSession).toHaveBeenCalledWith("old.jsonl"))
    } finally {
      omp.overview = null
    }
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

    const general = within(view.getByLabelText("general contents"))
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
    const shared = within(view.getByLabelText("Code"))
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
    expect(view.getByRole("status")).toHaveTextContent("Second workspace")
    fireEvent.keyDown(category, { key: "Tab", shiftKey: true })
    expect(view.getByRole("status")).toHaveTextContent("Second workspace")
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
      <TabManager ref={manager}><Sidebar appMode="chat" /><WorkspaceDraft /></TabManager>
    </BranchProvider>)
    const rootTab = { ...nextTab(), name: "Root", historyFile: "loose.json" }
    const childTab = { ...nextTab(), name: "Branch", historyFile: "child.json" }
    act(() => {
      manager.current!.openTab(rootTab)
      manager.current!.openTab(childTab)
    })
    fireEvent.change(view.getByRole("textbox", { name: "Draft Branch" }), { target: { value: "Keep this draft" } })
    const navigation = within(view.getByRole("navigation", { name: "Workspace" }))
    const general = within(await navigation.findByLabelText("general contents"))
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
    const general = within(await view.findByLabelText("general contents"))
    const history = await general.findByRole("button", { name: "loose" })
    const dataTransfer = { types: ["text/plain"], getData: (type: string) => type === "text/plain" ? "loose.json" : "" }
    fireEvent.drop(view.getByRole("button", { name: "Proj" }), { dataTransfer })
    expect(await view.findByRole("alert")).toHaveTextContent("Storage unavailable")
    expect(history).toBeVisible()
    expect(view.queryByLabelText("Proj contents")).not.toBeInTheDocument()
  })
})

describe("Sidebar tab list", () => {
  const activeRow = (view: RenderResult) => view.getByRole("button", { current: "page" }).textContent
  const rowNames = (view: RenderResult) => view.getAllByRole("button", { name: /^(?!Close ).* \/ / }).map((button) => button.textContent)
  beforeEach(() => localStorage.setItem("sidebar_tabs_expanded", "true"))

  it("starts collapsed with the open tab count and shares expansion across tabs", () => {
    const stored = new Map<string, string>()
    vi.stubGlobal("localStorage", { getItem: (key: string) => stored.get(key) ?? null, setItem: (key: string, value: string) => { stored.set(key, value) } })
    const manager = createRef<TabManagerHandle>()
    const view = render(<TabManager ref={manager}><TabList /></TabManager>)
    const first = manager.current!.getTabs()[0]
    act(() => manager.current!.openTab({ ...nextTab(), name: "Other" }))
    expect(view.getByRole("button", { name: "Tabs, 2 open" })).toHaveAttribute("aria-expanded", "false")
    expect(view.queryByRole("button", { name: "general / chats / Other" })).not.toBeInTheDocument()

    fireEvent.click(view.getByRole("button", { name: "Tabs, 2 open" }))
    act(() => manager.current!.switchToTab(first.id))
    expect(view.getByRole("button", { name: "Tabs, 2 open" })).toHaveAttribute("aria-expanded", "true")
    expect(activeRow(view)).toBe("general / chats / New chat")
  })

  it("switches on click, closes other tabs and never closes the last tab", () => {
    const manager = createRef<TabManagerHandle>()
    const view = render(<TabManager ref={manager}><TabList /></TabManager>)
    expect(view.queryByRole("button", { name: /^Close / })).not.toBeInTheDocument()
    act(() => manager.current!.openTab({ ...nextTab(), name: "Other" }))
    expect(activeRow(view)).toBe("general / chats / Other")

    fireEvent.click(view.getByRole("button", { name: "general / chats / New chat" }))
    expect(activeRow(view)).toBe("general / chats / New chat")
    fireEvent.click(view.getByRole("button", { name: "Close general / chats / Other" }))
    expect(manager.current!.getTabs()).toHaveLength(1)
    expect(view.queryByRole("button", { name: /^Close / })).not.toBeInTheDocument()
  })

  it("keeps Tab key focus traversal within the tree from switching workspaces", () => {
    const manager = createRef<TabManagerHandle>()
    const view = render(<TabManager ref={manager}><TabList /></TabManager>)
    act(() => manager.current!.openTab({ ...nextTab(), name: "Other" }))
    expect(fireEvent.keyDown(view.getByRole("button", { current: "page" }), { key: "Tab" })).toBe(true)
    expect(activeRow(view)).toBe("general / chats / Other")
  })

  it("reuses the selected tab for chats and items without disturbing background tabs", () => {
    const manager = createRef<TabManagerHandle>()
    const view = render(<TabManager ref={manager}><TabList /><WorkspaceDraft /></TabManager>)
    act(() => manager.current!.showChat("proj/first.json"))
    const background = manager.current!.getTabs()[0]
    const selected = nextTab()
    act(() => manager.current!.openTab(selected))
    fireEvent.change(view.getByRole("textbox"), { target: { value: "Unsaved draft" } })

    act(() => manager.current!.showChat("proj/first.json"))
    expect(activeRow(view)).toBe("Proj / chats / first.json")
    expect(view.getByRole("textbox")).toHaveValue("")
    fireEvent.change(view.getByRole("textbox"), { target: { value: "Session draft" } })
    act(() => manager.current!.showItem({ kind: "canvas", path: "board.canvas", projectSlug: "proj" }))
    expect(activeRow(view)).toBe("Proj / canvases / board.canvas")
    act(() => manager.current!.showChat("proj/first.json"))
    expect(view.getByRole("textbox")).toHaveValue("Session draft")

    act(() => manager.current!.showItem({ kind: "document", path: "/source/notes.md", projectSlug: null, rootFile: true }))
    expect(activeRow(view)).toBe("general / files / notes.md")
    act(() => manager.current!.showChat("proj/second.json"))
    expect(activeRow(view)).toBe("Proj / chats / second.json")
    expect(view.getByRole("textbox")).toHaveValue("")
    expect(manager.current!.getTabs().map((tab) => tab.id)).toEqual([background.id, selected.id])
    expect(manager.current!.getTabs()[0]).toEqual(background)
  })

  it("tracks the visible surface across project, shared, and scratch workspaces", () => {
    const manager = createRef<TabManagerHandle>()
    const view = render(<TabManager ref={manager}><TabList /></TabManager>)
    const canvas = { ...nextTab(), appMode: "canvas" as const, projectSlug: "proj", workspaceItem: { kind: "canvas" as const, path: "project/proj/document/board.canvas", projectSlug: "proj" } }
    const architecture = { ...nextTab(), appMode: "canvas" as const, projectSlug: "proj", workspaceItem: { kind: "architecture" as const, path: "graph/proj/", projectSlug: "proj" } }
    const file = { ...nextTab(), appMode: "document" as const, workspaceItem: { kind: "document" as const, path: "/source/notes.md", projectSlug: null, rootFile: true } }
    const scratch = { ...nextTab(), appMode: "canvas" as const, projectSlug: "proj" }
    act(() => {
      for (const tab of [canvas, architecture, file, scratch]) manager.current!.openTab(tab)
    })
    expect(activeRow(view)).toBe("Proj / canvases / new.canvas")
    for (const breadcrumb of ["Proj / canvases / board.canvas", "Proj / architecture / System overview", "general / files / notes.md"]) {
      fireEvent.click(view.getByRole("button", { name: breadcrumb }))
      expect(activeRow(view)).toBe(breadcrumb)
    }
    act(() => manager.current!.showChat("proj/session.json"))
    expect(activeRow(view)).toBe("Proj / chats / session.json")
  })

  it("names a project file's tab after its display name, other projects' tabs after the file", () => {
    const manager = createRef<TabManagerHandle>()
    const view = render(<TabManager ref={manager}><TabList /></TabManager>)
    const named = { ...nextTab(), appMode: "document" as const, projectSlug: "proj", workspaceItem: { kind: "document" as const, path: "project/proj/document/plan.md", projectSlug: "proj" } }
    const unnamed = { ...nextTab(), appMode: "document" as const, projectSlug: "empty", workspaceItem: { kind: "document" as const, path: "project/proj/document/plan.md", projectSlug: "empty" } }
    act(() => { for (const tab of [named, unnamed]) manager.current!.openTab(tab) })
    expect(view.getByRole("button", { name: "Proj / files / Roadmap.md" })).toBeInTheDocument()
    expect(view.getByRole("button", { name: "Empty project / files / plan.md" })).toBeInTheDocument()
  })

  it("sorts by project, category and natural filename order without reordering on selection", () => {
    const manager = createRef<TabManagerHandle>()
    const view = render(<TabManager ref={manager}><TabList /></TabManager>)
    const tabs = [
      { ...nextTab(), projectSlug: "proj", appMode: "document" as const, workspaceItem: { kind: "document" as const, path: "alpha.md", projectSlug: "proj" } },
      { ...nextTab(), projectSlug: "proj", appMode: "canvas" as const, workspaceItem: { kind: "canvas" as const, path: "board10.canvas", projectSlug: "proj" } },
      { ...nextTab(), projectSlug: "proj", title: "Alpha" },
      { ...nextTab(), projectSlug: "empty", title: "Z last" },
      { ...nextTab(), projectSlug: "proj", appMode: "canvas" as const, workspaceItem: { kind: "canvas" as const, path: "Board2.canvas", projectSlug: "proj" } },
      { ...nextTab(), projectSlug: "proj", appMode: "canvas" as const, workspaceItem: { kind: "architecture" as const, path: "graph/proj/", projectSlug: "proj" } },
    ]
    act(() => { for (const tab of tabs) manager.current!.openTab(tab) })
    const expected = [
      "Empty project / chats / Z last",
      "Proj / architecture / System overview",
      "Proj / canvases / Board2.canvas",
      "Proj / canvases / board10.canvas",
      "Proj / chats / Alpha",
      "Proj / files / alpha.md",
      "general / chats / New chat",
    ]
    expect(rowNames(view)).toEqual(expected)
    fireEvent.click(view.getByRole("button", { name: expected[3] }))
    expect(rowNames(view)).toEqual(expected)
    fireEvent.click(view.getByRole("button", { name: "Close Proj / files / alpha.md" }))
    expect(rowNames(view)).toEqual(expected.filter((label) => label !== "Proj / files / alpha.md"))
  })

  it("creates chats in the active workspace's project or general scope", () => {
    const manager = createRef<TabManagerHandle>()
    const view = render(<TabManager ref={manager}><TabList /></TabManager>)
    const projectTab = { ...nextTab(), name: "Project chat", projectSlug: "proj" }
    const sharedTab = { ...nextTab(), name: "Shared chat" }
    act(() => {
      manager.current!.openTab(projectTab)
      manager.current!.openTab(sharedTab)
    })
    for (const tab of [projectTab, sharedTab]) {
      act(() => manager.current!.switchToTab(tab.id))
      fireEvent.click(view.getByRole("button", { name: "New chat" }))
      expect(manager.current!.getTabs().at(-1)?.projectSlug).toBe(tab.projectSlug)
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
