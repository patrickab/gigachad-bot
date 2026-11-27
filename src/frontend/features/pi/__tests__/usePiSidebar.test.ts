import { renderHook } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { ApiError, type PiSessionInfo } from "@/lib/backend/api"
import { isInside } from "@/lib/paths"
import { piSessionError, piSessionForest, piSessionRepository, piSessionsInside, piSessionWorktree, unmountedPiWorkspaces, usePiSidebar } from "../usePiSidebar"

vi.mock("@/contexts/TabContext", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/contexts/TabContext")>(),
  useTab: () => ({ tab: { workspaceItem: null } }),
}))
const store = vi.hoisted(() => ({ sessions: [] as PiSessionInfo[], repositories: [] as { name: string; path: string }[] }))
vi.mock("@/lib/paths", async (importOriginal) => {
  const paths = await importOriginal<typeof import("@/lib/paths")>()
  return { ...paths, isInside: vi.fn(paths.isInside) }
})
vi.mock("../sessionStore", () => ({
  usePiSessions: () => ({ sessions: store.sessions, loading: false, error: null }),
  usePiRepositories: () => store.repositories,
  usePiSessionStatus: () => ({ running: false, waiting: false, unread: false, failed: false }),
  refreshPiSessions: () => {},
}))

const repositories = [{ name: "proj", path: "/git/proj" }, { name: "proj-two", path: "/git/proj-two" }, { name: "tool", path: "/git/tool" }]

function piSession(sessionId: string, cwd: string, extra: Partial<PiSessionInfo> = {}): PiSessionInfo {
  return {
    sessionId, sessionPath: `/s/${sessionId}.jsonl`, cwd, machine: "local", createdAt: "2026-01-01T00:00:00Z", modifiedAt: "2026-01-01T00:00:00Z",
    firstMessage: sessionId, messageCount: 1, tokens: 0, live: null, pinned: false, archived: false, ...extra,
  }
}

describe("Pi sidebar grouping", () => {
  it("puts a worktree session under its main checkout and names its branch", () => {
    const worktree = piSession("w", "/git/worktrees/proj/feat/src")
    expect(piSessionRepository(worktree, repositories)?.name).toBe("proj")
    expect(piSessionWorktree(worktree, repositories[0])).toBe("feat")
    expect(piSessionWorktree(piSession("m", "/git/proj"), repositories[0])).toBeNull()
    expect(piSessionRepository(piSession("x", "/git/proj-two"), repositories)?.name).toBe("proj-two")
  })

  it("lists a directory's sessions, worktrees included, and checkouts outside every mount with theirs", () => {
    const sessions = [
      piSession("main", "/git/proj/src"), piSession("tree", "/git/worktrees/proj/feat"), piSession("two", "/git/proj-two"),
      piSession("tool", "/git/tool"), piSession("home", "/home/me"),
    ]
    expect(piSessionsInside(sessions, "/git/proj", repositories).map((session) => session.sessionId)).toEqual(["main", "tree"])
    expect(unmountedPiWorkspaces(sessions, (session) => piSessionRepository(session, repositories), ["/git/proj"])).toEqual([
      { path: "/git/proj-two", label: "proj-two", sessions: [sessions[2]] },
      { path: "/git/tool", label: "tool", sessions: [sessions[3]] },
    ])
  })

  it("nests forks under their parent, newest first, and keeps orphans as roots", () => {
    const forest = piSessionForest([
      piSession("parent", "/git/proj", { modifiedAt: "2026-01-02T00:00:00Z" }),
      piSession("fork", "/git/proj", { parentSession: "/s/parent.jsonl" }),
      piSession("orphan", "/git/proj", { parentSession: "/s/gone.jsonl", modifiedAt: "2026-01-03T00:00:00Z" }),
    ])
    expect(forest.map((node) => [node.session.sessionId, node.forks.map((fork) => fork.session.sessionId)])).toEqual([["orphan", []], ["parent", ["fork"]]])
  })

  it("words pi-host's error codes for session actions and falls back for anything else", () => {
    expect(piSessionError(new ApiError("live", 409), "Could not archive this session")).toBe("Session is running")
    expect(piSessionError(new ApiError("not_found", 404), "Could not archive this session")).toBe("Session no longer exists")
    expect(piSessionError(new ApiError("toString", 409), "Could not archive this session")).toBe("Could not archive this session")
    expect(piSessionError(new Error("live"), "Could not pin this session")).toBe("Could not pin this session")
  })

  it("resolves each session's checkout once per list refresh, not once per directory", () => {
    store.repositories = repositories
    const directories = ["/git/proj", "/git/proj/a", "/git/proj/b", "/git/proj/c", "/git/proj/d"]
    const { result, rerender } = renderHook(() => usePiSidebar({}))
    store.sessions = Array.from({ length: 50 }, (_, index) => piSession(`s${index}`, `/git/proj/${"abcd"[index % 4]}`))
    vi.mocked(isInside).mockClear()
    rerender()
    for (const directory of directories) result.current.directoryCode(directory, "proj", null)
    const resolutions = vi.mocked(isInside).mock.calls.filter(([path, root]) => path !== root && root === "/git/proj").length
    expect(resolutions).toBe(50)
  })
})
