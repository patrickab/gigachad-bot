import { afterEach, expect, it, vi } from "vitest"
import { act, cleanup, render, waitFor } from "@testing-library/react"
import { BranchProvider, useBranches } from "@/features/navigation/BranchContext"
import { useResource } from "@/hooks/useResource"
import { projectsResource, watchDocumentLists } from "@/lib/backend/resources"

const api = vi.hoisted(() => ({
  listChatHistories: vi.fn(async () => ({ files: ["b.json", "a.json"] })),
  fetchBranchMeta: vi.fn(async () => ({ "a.json": { chat_id: "a" }, "b.json": { chat_id: "b", parent_id: "a" } })),
  listProjects: vi.fn(async () => []),
}))
vi.mock("@/lib/backend/api", async (importOriginal) => ({ ...await importOriginal<typeof import("@/lib/backend/api")>(), ...api }))
vi.mock("@/lib/backend/syncStream", () => ({ subscribeToChanges: () => () => {}, syncCursorReady: async () => {} }))

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

it("reads histories and branch metadata together, and a refresh reads the projects again", async () => {
  let branches: ReturnType<typeof useBranches> | undefined
  function Probe() {
    useResource("Projects", projectsResource)
    branches = useBranches()
    return null
  }
  const stop = watchDocumentLists()
  render(<BranchProvider onFileClick={() => {}}><Probe /></BranchProvider>)
  await waitFor(() => expect(branches?.historiesLoading).toBe(false))
  await waitFor(() => expect(projectsResource.get()).toEqual([]))
  expect(branches?.visibleRootFiles).toEqual(["a.json"])
  vi.clearAllMocks()

  await act(() => branches!.refreshAll())

  expect(api.listChatHistories).toHaveBeenCalledTimes(1)
  expect(api.fetchBranchMeta).toHaveBeenCalledTimes(1)
  await waitFor(() => expect(api.listProjects).toHaveBeenCalledTimes(1))
  stop()
})
