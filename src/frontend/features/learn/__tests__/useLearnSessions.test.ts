import { act, renderHook, waitFor } from "@testing-library/react"
import { afterEach, expect, it, vi } from "vitest"
import { ApiError } from "@/lib/backend/api"
import type { LearnSession } from "@/lib/domain/types"
import { useLearnSessions } from "../useLearnSessions"

const api = vi.hoisted(() => ({
  fetchLearnSessions: vi.fn<() => Promise<LearnSession[]>>(),
  assignLearnSession: vi.fn<(sessionId: string, projectSlug: string | null) => Promise<void>>(async () => {}),
  renameLearnSession: vi.fn<(sessionId: string, title: string) => Promise<string>>(),
  deleteLearnSession: vi.fn<(sessionId: string, mode?: string) => Promise<void>>(),
}))
vi.mock("@/lib/backend/api", async (importOriginal) => ({
  ApiError: (await importOriginal<typeof import("@/lib/backend/api")>()).ApiError,
  ...api,
}))

const session = (id: string, project_slug: string | null): LearnSession => ({ id, title: id, kind: "teach", busy: false, project_slug })

afterEach(() => {
  vi.resetAllMocks()
})

it("joins an unassigned session to the project its tab was opened from, never one that belongs to a project", async () => {
  let stored = [session("chat_new", null), session("chat_other", "maths")]
  api.fetchLearnSessions.mockImplementation(async () => stored)
  api.assignLearnSession.mockImplementation(async (id, slug) => {
    stored = stored.map((candidate) => candidate.id === id ? { ...candidate, project_slug: slug } : candidate)
  })
  const { result } = renderHook(() => useLearnSessions(true, false))
  await waitFor(() => expect(result.current.sessions).toHaveLength(2))

  await act(() => result.current.claim("chat_new", "physics"))
  await act(() => result.current.claim("chat_other", "physics"))
  await act(() => result.current.claim("chat_unknown", "physics"))

  expect(api.assignLearnSession.mock.calls).toEqual([["chat_new", "physics"]])
  expect(result.current.sessions.map((candidate) => candidate.project_slug)).toEqual(["physics", "maths"])
})

it("renames to the stored title, removes deleted sessions and keeps both lists on a refusal", async () => {
  api.fetchLearnSessions.mockResolvedValue([session("chat_a", null), session("chat_b", "maths")])
  const { result } = renderHook(() => useLearnSessions(true, false))
  await waitFor(() => expect(result.current.sessions).toHaveLength(2))

  api.renameLearnSession.mockResolvedValueOnce("Fourier series")
  await act(() => result.current.rename("chat_a", "  Fourier series "))
  expect(api.renameLearnSession).toHaveBeenCalledWith("chat_a", "  Fourier series ")
  expect(result.current.sessions.map((candidate) => candidate.title)).toEqual(["Fourier series", "chat_b"])

  api.deleteLearnSession.mockRejectedValueOnce(new ApiError("A turn is running", 409))
  await expect(act(() => result.current.remove("chat_b"))).rejects.toThrow("A turn is running")
  expect(result.current.sessions).toHaveLength(2)

  expect(result.current.missing.has("chat_b")).toBe(false)

  await act(() => result.current.remove("chat_b"))
  expect(result.current.sessions.map((candidate) => candidate.id)).toEqual(["chat_a"])
  expect(result.current.missing.has("chat_b")).toBe(true)

  await act(() => result.current.remove("chat_a", "delete-nodes"))
  expect(api.deleteLearnSession).toHaveBeenLastCalledWith("chat_a", "delete-nodes")
  expect(result.current.sessions).toEqual([])
})
