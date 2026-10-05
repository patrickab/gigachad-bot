/**
 * The canvas tree is the only way to open a saved canvas: a click must report the
 * canvas's path and owning scope. It once lost its click handler and listed every
 * canvas without being able to open any.
 */
import { describe, it, expect, vi } from "vitest"
import { render, fireEvent, waitFor } from "@testing-library/react"
import { Sidebar } from "@/components/Sidebar"

const api = vi.hoisted(() => ({
  listNotes: vi.fn(async () => [{ path: "chat_histories/_notes/loose.canvas", name: "loose.canvas", mime: "text/plain" }]),
  listProjectDocuments: vi.fn(async () => [{ path: "project/proj/document/board.canvas", name: "board.canvas", mime: "text/plain" }]),
}))
vi.mock("@/lib/api", () => ({
  ...api,
  addFileVaultMountpoint: vi.fn(), addFileVaultRoot: vi.fn(), createDirectory: vi.fn(), moveHistoryItem: vi.fn(),
  fileVaultTree: vi.fn(async () => ({ tree: [] })), removeFileVaultRoot: vi.fn(), moveDocument: vi.fn(),
  removeDocument: vi.fn(), writeDocument: vi.fn(), Vault: class {},
}))
vi.mock("@/contexts/SidebarContext", () => ({
  useSidebar: () => ({
    collapsed: false, toggleCollapsed: vi.fn(), projectsOpen: true, setProjectsOpen: vi.fn(), historiesOpen: true, setHistoriesOpen: vi.fn(),
  }),
}))
vi.mock("@/contexts/BranchContext", () => ({
  useBranches: () => ({
    branchMeta: {}, visibleRootFiles: [], visibleHistories: [], historiesLoading: false,
    registerOnFileClick: vi.fn(), registerOnMerge: vi.fn(), registerOnDelete: vi.fn(),
  }),
}))
vi.mock("@/contexts/ProjectContext", () => ({
  useProject: () => ({
    projects: [{ slug: "proj", name: "Proj", tabs: [] }], activeProject: null, openProject: vi.fn(), closeProject: vi.fn(),
    createProject: vi.fn(), deleteProject: vi.fn(), setDashboardOpen: vi.fn(),
  }),
}))
vi.mock("@/contexts/MemoryViewerContext", () => ({ useMemoryViewer: () => ({ openMemoryViewer: vi.fn() }) }))
vi.mock("@/contexts/UndoDeleteContext", () => ({ useUndoDelete: () => ({ schedule: vi.fn(), pending: () => false }) }))

describe("Sidebar canvas tree", () => {
  it("opens a project canvas and a loose canvas with the scope that owns them", async () => {
    const onCanvasSelect = vi.fn()
    const { findByText, getByText } = render(
      <Sidebar
        onOpenChat={vi.fn()} onRefreshAll={vi.fn(async () => {})} onSave={vi.fn()} onReset={vi.fn()}
        appMode="canvas" onAppModeChange={vi.fn()} onCanvasSelect={onCanvasSelect}
      />,
    )

    fireEvent.click(await findByText("loose"))
    expect(onCanvasSelect).toHaveBeenLastCalledWith("chat_histories/_notes/loose.canvas", "")

    // Project canvases sit inside their (collapsed) project row.
    fireEvent.click(getByText("Proj"))
    fireEvent.click(await findByText("board"))
    await waitFor(() => expect(onCanvasSelect).toHaveBeenLastCalledWith("project/proj/document/board.canvas", "proj"))
  })
})
