/** The dashboard loads and mutates exactly the project it is given, never a globally active one. */
import { act, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, expect, it, vi } from "vitest"
import { UndoDeleteProvider } from "@/contexts/UndoDeleteContext"
import { ProjectDashboard } from "../ProjectDashboard"

const api = vi.hoisted(() => ({
  loadProject: vi.fn(),
  addProjectCard: vi.fn(),
  moveProjectCard: vi.fn(),
  deleteProjectCard: vi.fn(),
}))
vi.mock("@/lib/backend/api", () => api)

const card = { id: "c1", title: "Write spec", description: "", state: "backlog" }

function renderDashboard(slug: string, onClose = vi.fn()) {
  render(<UndoDeleteProvider><ProjectDashboard slug={slug} onClose={onClose} /></UndoDeleteProvider>)
  return onClose
}

beforeEach(() => {
  vi.clearAllMocks()
  api.loadProject.mockImplementation(async (slug: string) => ({ name: `Name of ${slug}`, slug, kanban: [card], tabs: [] }))
  api.moveProjectCard.mockImplementation(async (_slug: string, id: string, state: string) => ({ ...card, id, state }))
})

it("does not replace a newly opened board with a previous project's late response", async () => {
  let finishAlpha!: (data: unknown) => void
  api.loadProject.mockImplementationOnce(() => new Promise((resolve) => { finishAlpha = resolve }))
  const view = render(<UndoDeleteProvider><ProjectDashboard key="alpha" slug="alpha" onClose={vi.fn()} /></UndoDeleteProvider>)
  view.rerender(<UndoDeleteProvider><ProjectDashboard key="beta" slug="beta" onClose={vi.fn()} /></UndoDeleteProvider>)
  expect(await screen.findByText("Name of beta Dashboard")).toBeInTheDocument()
  await act(async () => { finishAlpha({ name: "Alpha", slug: "alpha", kanban: [], tabs: [] }) })
  expect(screen.queryByText("Alpha Dashboard")).not.toBeInTheDocument()
  expect(screen.getByText("Name of beta Dashboard")).toBeInTheDocument()
  expect(screen.getByText("Write spec")).toBeInTheDocument()
})

it("closes itself when the project cannot be loaded", async () => {
  api.loadProject.mockRejectedValueOnce(new Error("missing"))
  const onClose = renderDashboard("gone")
  await waitFor(() => expect(onClose).toHaveBeenCalledOnce())
})
