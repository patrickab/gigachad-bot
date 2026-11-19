import { describe, expect, it } from "vitest"
import type { PiSessionInfo } from "@/lib/backend/api"
import { piSessionForest, piSessionRepository, piSessionsInside, piSessionTitle, piSessionWorktree, piStatusMark, piUsageText, unmountedPiWorkspaces } from "../usePiSidebar"

const repositories = [{ name: "proj", path: "/git/proj" }, { name: "proj-two", path: "/git/proj-two" }, { name: "tool", path: "/git/tool" }]

function piSession(sessionId: string, cwd: string, extra: Partial<PiSessionInfo> = {}): PiSessionInfo {
  return {
    sessionId, sessionPath: `/s/${sessionId}.jsonl`, cwd, machine: "local", createdAt: "2026-01-01T00:00:00Z", modifiedAt: "2026-01-01T00:00:00Z",
    firstMessage: sessionId, messageCount: 1, live: null, pinned: false, archived: false, ...extra,
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
})
