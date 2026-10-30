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
  vi.useRealTimers()
})

/** Lets the first read and the reads a timer starts settle. */
async function settle(ms = 0) {
  await act(async () => { await vi.advanceTimersByTimeAsync(ms) })
}

it("reads nothing for logins without learn", async () => {
  const { result } = renderHook(() => useLearnSessions(false, true))
  await act(async () => {})
  expect(api.fetchLearnSessions).not.toHaveBeenCalled()
  expect(result.current.sessions).toEqual([])
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

it("keeps the last list and reports a failed read", async () => {
  api.fetchLearnSessions.mockResolvedValueOnce([session("chat_a", null)])
  const { result } = renderHook(() => useLearnSessions(true, false))
  await waitFor(() => expect(result.current.sessions).toHaveLength(1))

  api.fetchLearnSessions.mockRejectedValueOnce(new Error("LearnHarness answered without the field id"))
  await act(() => result.current.refresh())

  expect(result.current.sessions).toEqual([session("chat_a", null)])
  expect(result.current.error).toMatch(/without the field id/)
})

it("polls only while the frame is watched or a session is busy", async () => {
  vi.useFakeTimers()
  api.fetchLearnSessions.mockResolvedValue([session("chat_a", null)])
  const { rerender } = renderHook(({ watching }) => useLearnSessions(true, watching), { initialProps: { watching: false } })
  await settle()
  expect(api.fetchLearnSessions).toHaveBeenCalledTimes(1)

  await settle(60_000)
  expect(api.fetchLearnSessions).toHaveBeenCalledTimes(1)

  // A watched frame polls; the read finds a busy session, which keeps polling once the frame is left.
  api.fetchLearnSessions.mockResolvedValue([{ ...session("chat_a", null), busy: true }])
  rerender({ watching: true })
  await settle(15_000)
  expect(api.fetchLearnSessions).toHaveBeenCalledTimes(2)

  rerender({ watching: false })
  await settle(15_000)
  expect(api.fetchLearnSessions).toHaveBeenCalledTimes(3)

  api.fetchLearnSessions.mockResolvedValue([session("chat_a", null)])
  await settle(15_000)
  await settle(60_000)
  expect(api.fetchLearnSessions).toHaveBeenCalledTimes(4)
})

it("stays quiet and stops polling while LearnHarness is unavailable, until the window is focused again", async () => {
  vi.useFakeTimers()
  api.fetchLearnSessions.mockRejectedValue(new ApiError("LearnHarness at http://127.0.0.1:30181 does not answer", 503))
  const { result } = renderHook(() => useLearnSessions(true, true))
  await settle()
  expect(result.current.error).toBeNull()

  await settle(60_000)
  expect(api.fetchLearnSessions).toHaveBeenCalledTimes(1)

  api.fetchLearnSessions.mockResolvedValue([session("chat_a", null)])
  await act(async () => { window.dispatchEvent(new Event("focus")) })
  await settle(15_000)
  expect(api.fetchLearnSessions).toHaveBeenCalledTimes(3)
  expect(result.current.sessions).toEqual([session("chat_a", null)])
})

it("looks a session up once, and marks it missing when a read lacks it", async () => {
  api.fetchLearnSessions.mockResolvedValue([session("chat_a", null)])
  const { result } = renderHook(() => useLearnSessions(true, false))
  await waitFor(() => expect(result.current.sessions).toHaveLength(1))
  api.fetchLearnSessions.mockClear()

  await act(() => result.current.lookUp("chat_gone"))
  await act(() => result.current.lookUp("chat_gone"))

  expect(api.fetchLearnSessions).toHaveBeenCalledTimes(1)
  expect(result.current.missing.has("chat_gone")).toBe(true)
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

it("retitles only a listed session whose title differs, keeping the list otherwise", async () => {
  api.fetchLearnSessions.mockResolvedValue([session("chat_a", null)])
  const { result } = renderHook(() => useLearnSessions(true, false))
  await waitFor(() => expect(result.current.sessions).toHaveLength(1))
  const listed = result.current.sessions

  act(() => result.current.retitle("chat_a", "chat_a"))
  act(() => result.current.retitle("chat_unknown", "Other"))
  expect(result.current.sessions).toBe(listed)

  act(() => result.current.retitle("chat_a", "Renamed in the frame"))
  expect(result.current.sessions[0].title).toBe("Renamed in the frame")
})
