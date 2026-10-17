/**
 * Opening a blank tab (or changing a tab's model) used to rewrite the open project's
 * state and refetch the project list, re-rendering the whole project tree.
 */
import { act, renderHook } from "@testing-library/react"
import { expect, it, vi } from "vitest"
import type { ReactNode } from "react"
import { ProjectProvider, useProject } from "../ProjectContext"

const api = vi.hoisted(() => ({
  listProjects: vi.fn(async () => []),
  loadProject: vi.fn(async () => ({ name: "Proj", slug: "proj", kanban: [], tabs: [{ filename: "saved.json", name: null, title: "Saved" }] })),
  updateProjectKanban: vi.fn(async (_slug: string, update: { kanban: []; tabs: unknown[] }) => ({ name: "Proj", slug: "proj", ...update })),
}))
vi.mock("@/lib/api", () => ({
  ...api,
  createProject: vi.fn(), deleteProject: vi.fn(), addProjectCard: vi.fn(), moveProjectCard: vi.fn(), deleteProjectCard: vi.fn(),
}))

const wrapper = ({ children }: { children: ReactNode }) => <ProjectProvider>{children}</ProjectProvider>
const savedTab = { id: "a", name: null, historyFile: "proj/saved.json", title: "Saved", chatId: "1" }
const blankTab = { id: "b", name: null, historyFile: null, title: null, chatId: "2" }

it("writes project state only when the saved tabs change", async () => {
  const { result } = renderHook(() => useProject(), { wrapper })
  await act(async () => { await result.current.openProject("proj") })

  await act(async () => { await result.current.syncTabs([savedTab, blankTab]) })
  expect(api.updateProjectKanban).not.toHaveBeenCalled()

  const renamed = { ...savedTab, title: "Renamed" }
  await act(async () => { await result.current.syncTabs([renamed, blankTab]) })
  expect(api.updateProjectKanban).toHaveBeenCalledTimes(1)
  expect(api.updateProjectKanban).toHaveBeenLastCalledWith("proj", { kanban: [], tabs: [{ filename: "saved.json", name: null, title: "Renamed" }] })

  await act(async () => { await result.current.syncTabs([renamed]) })
  expect(api.updateProjectKanban).toHaveBeenCalledTimes(1)
})
