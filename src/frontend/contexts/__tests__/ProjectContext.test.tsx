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
  renameProject: vi.fn(async (_slug: string, name: string) => ({ name, slug: "renamed", archived: false })),
}))
vi.mock("@/lib/backend/api", () => api)
const sync = vi.hoisted(() => ({ receive: null as null | ((event: { resource_kind: string; resource_key: string }) => void) }))
vi.mock("@/lib/backend/syncStream", () => ({ subscribeToChanges: (receive: typeof sync.receive) => { sync.receive = receive; return () => { sync.receive = null } } }))

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

it("renames a project and returns the slug the server moved it to", async () => {
  api.listProjects.mockResolvedValue([{ name: "Renamed", slug: "renamed" }]).mockResolvedValueOnce([{ name: "Original", slug: "original" }])
  const { result } = renderHook(() => useProject(), { wrapper })
  await waitFor(() => expect(result.current.projects[0]?.slug).toBe("original"))
  let slug = ""
  await act(async () => { slug = await result.current.renameProject("original", "Renamed") })
  expect(slug).toBe("renamed")
  expect(result.current.projects).toEqual([{ name: "Renamed", slug: "renamed" }])
})

it("refreshes the list when another device changes the project catalog", async () => {
  api.listProjects.mockResolvedValue([{ name: "Renamed", slug: "renamed" }]).mockResolvedValueOnce([{ name: "Original", slug: "original" }])
  const { result } = renderHook(() => useProject(), { wrapper })
  await waitFor(() => expect(result.current.projects[0]?.slug).toBe("original"))
  await act(async () => { sync.receive?.({ resource_kind: "document", resource_key: "chat/original/project.json" }) })
  expect(result.current.projects[0].slug).toBe("original")
  await act(async () => { sync.receive?.({ resource_kind: "document", resource_key: "chat/projects-meta.json" }) })
  expect(result.current.projects[0].slug).toBe("renamed")
})
