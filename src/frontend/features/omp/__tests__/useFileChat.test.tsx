import { afterEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, renderHook, waitFor } from "@testing-library/react"

const api = vi.hoisted(() => ({
  loadFileSessions: vi.fn<(path: string) => Promise<{ active: string | null; sessionIds: string[] }>>(),
  saveFileSession: vi.fn(async (_path: string, sessionId: string | null) => ({ active: sessionId, sessionIds: sessionId ? [sessionId] : [] })),
}))
vi.mock("@/lib/backend/api", () => api)

import { useFileChat } from "../useFileChat"

const PATH = "/data/workbench/markdown/proj/plan/plan.md"
const CWD = "/data/workbench/markdown"
const sessions = (active: string | null, sessionIds: string[] = active ? [active] : []) => ({ active, sessionIds })

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe("useFileChat", () => {
  it("shows the file's active session, else a new one in cwd", async () => {
    api.loadFileSessions.mockResolvedValueOnce(sessions("linked"))
    const linked = renderHook(() => useFileChat(PATH, CWD, undefined, vi.fn()))
    expect(linked.result.current.view).toBeNull()
    await waitFor(() => expect(linked.result.current.view).toEqual({ kind: "session", sessionId: "linked" }))
    expect(api.loadFileSessions).toHaveBeenCalledWith(PATH)

    api.loadFileSessions.mockResolvedValueOnce(sessions(null, ["earlier"]))
    const fresh = renderHook(() => useFileChat(PATH, CWD, undefined, vi.fn()))
    await waitFor(() => expect(fresh.result.current.view).toEqual({ kind: "new", cwd: CWD }))
    expect(fresh.result.current.sessionIds).toEqual(["earlier"])
  })

  it("keeps the item's session over the file's active one and still lists the file's sessions", async () => {
    api.loadFileSessions.mockResolvedValueOnce(sessions("linked", ["linked", "shown"]))
    const { result } = renderHook(() => useFileChat(PATH, CWD, { kind: "session", sessionId: "shown" }, vi.fn()))
    expect(result.current.view).toEqual({ kind: "session", sessionId: "shown" })
    await waitFor(() => expect(result.current.sessionIds).toEqual(["linked", "shown"]))
    expect(result.current.view).toEqual({ kind: "session", sessionId: "shown" })
  })

  it("links an adopted session once and hands it to the item", async () => {
    api.loadFileSessions.mockResolvedValueOnce(sessions(null))
    const onViewChange = vi.fn()
    const { result } = renderHook(() => useFileChat(PATH, CWD, undefined, onViewChange))
    await waitFor(() => expect(result.current.view).not.toBeNull())

    act(() => result.current.adopt({ kind: "session", sessionId: "started" }))
    act(() => result.current.adopt({ kind: "session", sessionId: "started" }))
    expect(api.saveFileSession).toHaveBeenCalledExactlyOnceWith(PATH, "started")
    expect(onViewChange).toHaveBeenLastCalledWith({ kind: "session", sessionId: "started" })
    await waitFor(() => expect(result.current.sessionIds).toEqual(["started"]))
  })

  it("resets to a new session and reopens an earlier one, saving each choice as active", async () => {
    api.loadFileSessions.mockResolvedValueOnce(sessions("first", ["first"]))
    api.saveFileSession.mockResolvedValueOnce(sessions(null, ["first"])).mockResolvedValueOnce(sessions("first", ["first"]))
    const onViewChange = vi.fn()
    const { result } = renderHook(() => useFileChat(PATH, CWD, undefined, onViewChange))
    await waitFor(() => expect(result.current.view).toEqual({ kind: "session", sessionId: "first" }))

    act(() => result.current.reset())
    expect(onViewChange).toHaveBeenLastCalledWith({ kind: "new", cwd: CWD })
    expect(api.saveFileSession).toHaveBeenLastCalledWith(PATH, null)

    act(() => result.current.select("first"))
    expect(onViewChange).toHaveBeenLastCalledWith({ kind: "session", sessionId: "first" })
    expect(api.saveFileSession).toHaveBeenLastCalledWith(PATH, "first")
  })

  it("says why the links could not load", async () => {
    api.loadFileSessions.mockRejectedValueOnce(new Error("offline"))
    const { result } = renderHook(() => useFileChat(PATH, CWD, undefined, vi.fn()))
    await waitFor(() => expect(result.current.error).toBe("offline"))
    expect(result.current.view).toBeNull()
  })
})
