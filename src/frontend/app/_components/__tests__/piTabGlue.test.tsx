import { afterEach, describe, expect, it, vi } from "vitest"
import { act, renderHook, waitFor } from "@testing-library/react"
import type { Tab, WorkspaceItem } from "@/contexts/TabContext"
import { fetchPiRepositories } from "@/lib/backend/api"
import type { Message } from "@/lib/domain/types"
import { usePiFolder } from "../TabContent"
import { useMemorizeSource } from "../useTabChat"

const { history } = vi.hoisted(() => ({ history: vi.fn() }))
vi.mock("@/features/pi/PiTab", () => ({ piSessionHistory: history }))
vi.mock("@/lib/backend/api", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/lib/backend/api")>(),
  fetchPiRepositories: vi.fn(),
}))

const CHAT: Message[] = [{ role: "user", content: "hidden chat" }]

function piTab(sessionId: string | null): Tab {
  const item: WorkspaceItem = { kind: "pi", path: "repo", projectSlug: null, pi: { sessionId, repo: "repo" } }
  return { id: "t", name: null, chatId: "c", historyFile: null, title: null, config: {} as Tab["config"], appMode: "pi", projectSlug: null, workspaceItem: item }
}

describe("useMemorizeSource", () => {
  afterEach(() => vi.clearAllMocks())

  it("reads a Pi tab's session, never the hidden chat", async () => {
    history.mockReturnValue([{ role: "user", content: "fix it" }])
    const { result } = renderHook(() => useMemorizeSource(piTab("s1"), CHAT, "c", null))
    await expect(result.current?.()).resolves.toEqual({ messages: [{ role: "user", content: "fix it" }], conversationId: "s1" })
    expect(history).toHaveBeenCalledWith("s1")
  })

  it("has nothing to memorize in a Pi tab without a session", () => {
    const { result } = renderHook(() => useMemorizeSource(piTab(null), CHAT, "c", null))
    expect(result.current).toBeNull()
  })

  it("fails while the session's view is not open", async () => {
    history.mockReturnValue(null)
    const { result } = renderHook(() => useMemorizeSource(piTab("s1"), CHAT, "c", null))
    await expect(result.current?.()).rejects.toThrow("The Pi session is not open yet")
  })
})

describe("usePiFolder", () => {
  afterEach(() => vi.clearAllMocks())

  it("takes a checkout under ~/git", async () => {
    vi.mocked(fetchPiRepositories).mockResolvedValue([{ name: "repo", path: "/home/u/git/repo" }])
    const { result } = renderHook(() => usePiFolder())
    act(() => result.current.pickFolder("/home/u/git/repo/"))
    await waitFor(() => expect(result.current.folder).toBe("/home/u/git/repo"))
    expect(result.current.folderError).toBeNull()
  })

  it("refuses a mounted folder outside ~/git with a clear error", async () => {
    vi.mocked(fetchPiRepositories).mockResolvedValue([{ name: "repo", path: "/home/u/git/repo" }])
    const { result } = renderHook(() => usePiFolder())
    act(() => result.current.pickFolder("/mnt/notes"))
    await waitFor(() => expect(result.current.folderError).toBe("notes is not a Git checkout under ~/git, Pi sessions start only there"))
    expect(result.current.folder).toBeNull()
  })

  it("keeps the latest of overlapping picks", async () => {
    const first = Promise.withResolvers<{ name: string; path: string }[]>()
    vi.mocked(fetchPiRepositories)
      .mockReturnValueOnce(first.promise)
      .mockResolvedValueOnce([{ name: "b", path: "/home/u/git/b" }])
    const { result } = renderHook(() => usePiFolder())
    act(() => result.current.pickFolder("/home/u/git/a"))
    act(() => result.current.pickFolder("/home/u/git/b"))
    await waitFor(() => expect(result.current.folder).toBe("/home/u/git/b"))
    await act(async () => first.resolve([{ name: "a", path: "/home/u/git/a" }]))
    expect(result.current.folder).toBe("/home/u/git/b")
  })
})
