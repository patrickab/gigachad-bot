/**
 * The provider owns only the project list. Creating or deleting a project refreshes it;
 * nothing is loaded, opened or closed, because each chat uses its own tab's `projectSlug`.
 */
import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, expect, it, vi } from "vitest"
import type { ReactNode } from "react"
import { ProjectProvider, useProject } from "../ProjectContext"

const api = vi.hoisted(() => ({
  listProjects: vi.fn(async () => [] as { name: string; slug: string }[]),
  createProject: vi.fn(async (name: string) => ({ name, slug: "new-project", kanban: [], tabs: [] })),
  deleteProject: vi.fn(async () => undefined),
  loadProject: vi.fn(),
}))
vi.mock("@/lib/backend/api", () => api)

const wrapper = ({ children }: { children: ReactNode }) => <ProjectProvider>{children}</ProjectProvider>

beforeEach(() => { vi.clearAllMocks() })

it("creates a project and refreshes the list without loading it", async () => {
  api.listProjects.mockResolvedValueOnce([]).mockResolvedValueOnce([{ name: "New project", slug: "new-project" }])
  const { result } = renderHook(() => useProject(), { wrapper })
  await waitFor(() => expect(api.listProjects).toHaveBeenCalledTimes(1))

  let slug = ""
  await act(async () => { slug = await result.current.createProject("New project") })

  expect(slug).toBe("new-project")
  expect(result.current.projects).toEqual([{ name: "New project", slug: "new-project" }])
  expect(api.loadProject).not.toHaveBeenCalled()
})

it("deletes a project and refreshes the list", async () => {
  api.listProjects.mockResolvedValueOnce([{ name: "Proj", slug: "proj" }]).mockResolvedValueOnce([])
  const { result } = renderHook(() => useProject(), { wrapper })
  await waitFor(() => expect(result.current.projects).toHaveLength(1))

  await act(async () => { await result.current.deleteProject("proj") })

  expect(api.deleteProject).toHaveBeenCalledWith("proj")
  expect(result.current.projects).toEqual([])
  expect(api.loadProject).not.toHaveBeenCalled()
})
