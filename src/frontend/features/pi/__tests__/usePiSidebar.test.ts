import { describe, expect, it, vi } from "vitest"
import { renderHook } from "@testing-library/react"
import { ApiError, type PiSessionInfo } from "@/lib/backend/api"
import { piSessionError, piSessionForest, piSessionRepository, piSessionsInside, piSessionTitle, piSessionWorktree, piStatusMark, piTokenText, piUsageText, unmountedPiWorkspaces, usePiSidebar } from "../usePiSidebar"

const store = vi.hoisted(() => ({ listFilters: [] as unknown[] }))
vi.mock("@/contexts/TabContext", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/contexts/TabContext")>(),
  useTab: () => ({ tab: { workspaceItem: null } }),
}))
vi.mock("../sessionStore", () => ({
  usePiSessions: (filter: unknown) => {
    store.listFilters.push(filter)
    return { sessions: [], loading: false, error: null }
  },
  usePiRepositories: () => [],
  usePiSessionStatus: () => ({ running: false, waiting: false, unread: false }),
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
    expect(unmountedPiWorkspaces(sessions, repositories, ["/git/proj"])).toEqual([
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

  it("titles a session by its first prompt line and marks its state", () => {
    expect(piSessionTitle({ firstMessage: "  fix the tabs\nmore" })).toBe("fix the tabs")
    expect(piSessionTitle({ firstMessage: "" })).toBe("Untitled session")
    expect(piStatusMark({ running: true, waiting: true, unread: false })).toEqual({ tone: "attention", label: "Waiting" })
    expect(piStatusMark({ running: true, waiting: false, unread: false })).toEqual({ tone: "active", label: "Running" })
    expect(piStatusMark({ running: false, waiting: false, unread: true })).toEqual({ tone: "done", label: "Unread" })
    expect(piStatusMark({ running: false, waiting: false, unread: false })).toBeNull()
    expect(piUsageText({ contextPercent: 12.4, cost: 1.5 })).toBe("12% · $1.50")
    expect(piUsageText({})).toBe("")
  })

  it("words pi-host's error codes for session actions and falls back for anything else", () => {
    expect(piSessionError(new ApiError("live", 409), "Could not archive this session")).toBe("Session is running")
    expect(piSessionError(new ApiError("not_found", 404), "Could not archive this session")).toBe("Session no longer exists")
    expect(piSessionError(new ApiError("toString", 409), "Could not archive this session")).toBe("Could not archive this session")
    expect(piSessionError(new Error("live"), "Could not pin this session")).toBe("Could not pin this session")
  })
})

describe("usePiSidebar", () => {
  it("lists only coding sessions (G5)", () => {
    renderHook(() => usePiSidebar({}))
    expect(store.listFilters).toContainEqual({ profile: "coding" })
    for (const filter of store.listFilters) expect(filter).toMatchObject({ profile: "coding" })
  })
})

describe("piTokenText", () => {
  it("shows the context size in thousands, nothing before the first answer", () => {
    expect(piTokenText(0)).toBe("")
    expect(piTokenText(400)).toBe("<1k")
    expect(piTokenText(84_300)).toBe("84k")
  })
})
