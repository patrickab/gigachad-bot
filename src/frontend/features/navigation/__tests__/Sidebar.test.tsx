/**
 * The canvas tree is the only way to open a saved canvas: a click must report the
 * canvas's path and owning scope. It once lost its click handler and listed every
 * canvas without being able to open any.
 */
import { describe, it, expect, vi } from "vitest"
import { render, fireEvent, waitFor } from "@testing-library/react"
import { Sidebar } from "@/features/navigation/Sidebar"

const api = vi.hoisted(() => ({
  listNotes: vi.fn(async () => [{ path: "chat_histories/_notes/loose.canvas", name: "loose.canvas", mime: "text/plain" }]),
  listProjectDocuments: vi.fn(async () => [
    { path: "project/proj/document/board.canvas", name: "board.canvas", mime: "text/plain" },
    { path: "graph/proj/backend/backend.c4", name: "backend/backend.c4", mime: "text/plain" },
  ]),
}))
vi.mock("@/lib/backend/api", () => ({
  ...api,
  addFileRoot: vi.fn(), moveHistoryItem: vi.fn(),
  fileRootTree: vi.fn(async () => ({ tree: [] })), removeFileRoot: vi.fn(), moveDocument: vi.fn(),
  removeDocument: vi.fn(), writeDocument: vi.fn(),
}))
vi.mock("@/features/navigation/SidebarContext", () => ({
  useSidebar: () => ({
    collapsed: false,
  }),
}))
vi.mock("@/features/navigation/BranchContext", () => ({
  useBranches: () => ({
    branchMeta: {}, visibleRootFiles: [], historiesLoading: false, refreshAll: vi.fn(async () => {}),
  }),
}))
vi.mock("@/contexts/ProjectContext", () => ({
  useProject: () => ({
    projects: [{ slug: "proj", name: "Proj", tabs: [] }], activeProject: null, openProject: vi.fn(), closeProject: vi.fn(),
    createProject: vi.fn(), deleteProject: vi.fn(), setDashboardOpen: vi.fn(),
  }),
}))
vi.mock("@/features/memory/MemoryViewerContext", () => ({ useMemoryViewer: () => ({ openMemoryViewer: vi.fn() }) }))
vi.mock("@/contexts/UndoDeleteContext", () => ({ useUndoDelete: () => ({ schedule: vi.fn(), pending: () => false }) }))

describe("Sidebar canvas tree", () => {
  it("opens a project's architecture, a project canvas and a loose canvas with the scope that owns them", async () => {
    const onCanvasSelect = vi.fn()
    const onArchitectureSelect = vi.fn()
    const { findByText, getByText } = render(
      <Sidebar
        appMode="canvas" onAppModeChange={vi.fn()} onCanvasSelect={onCanvasSelect} onArchitectureSelect={onArchitectureSelect}
      />,
    )

    fireEvent.click(await findByText("loose"))
    expect(onCanvasSelect).toHaveBeenLastCalledWith("chat_histories/_notes/loose.canvas", "")

    // A project's architecture and canvases sit inside its (collapsed) project row.
    fireEvent.click(getByText("Proj"))
    fireEvent.click(await findByText("Architecture"))
    expect(onArchitectureSelect).toHaveBeenLastCalledWith("proj")
    fireEvent.click(getByText("board"))
    await waitFor(() => expect(onCanvasSelect).toHaveBeenLastCalledWith("project/proj/document/board.canvas", "proj"))
  })
})
