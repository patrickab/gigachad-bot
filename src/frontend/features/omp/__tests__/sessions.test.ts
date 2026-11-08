import { describe, expect, it } from "vitest"
import { RECENT_SESSION_MS, groupSessions, historyPairs, projectMount, sessionForest, sessionsInside, startDirectories, unlinkedWorkspaces } from "@/features/omp/sessions"
import type { OmpOverview, OmpSession } from "@/lib/domain/types"

function session(id: string, cwd: string, extra: Partial<OmpSession> = {}): OmpSession {
  return {
    id, path: `/s/${id}.jsonl`, cwd, name: null, created: "2026-01-01", modified: "2026-01-01", message_count: 1,
    first_message: id, parent_session_id: null, project_root: null, worktree_branch: null, ...extra,
  }
}

describe("historyPairs", () => {
  it("joins an answer split over tool steps, pads an unanswered prompt and a reply without a prompt", () => {
    const pairs = historyPairs([
      { role: "assistant", content: "greeting" },
      { role: "user", content: "q1" },
      { role: "user", content: "q2" },
      { role: "assistant", content: "a" },
      { role: "assistant", content: "b" },
    ])
    expect(pairs.map(({ user, assistant, globalIndex, qaIndex }) => [user.content, assistant.content, globalIndex, qaIndex])).toEqual([
      ["", "greeting", 0, 0],
      ["q1", "", 1, 1],
      ["q2", "a\n\nb", 2, 2],
    ])
  })
})

describe("project code sessions", () => {
  const mounts = ["/home/me/git/other", "/home/me/git/gigachad-bot/"]

  it("links a project to the mount named like its slug", () => {
    expect(projectMount("gigachad-bot", mounts)).toBe("/home/me/git/gigachad-bot/")
    expect(projectMount("missing", mounts)).toBeNull()
  })

  it("collects sessions in subdirectories and in worktrees of the linked mount", () => {
    const sessions = [
      session("sub", "/home/me/git/gigachad-bot/src"),
      session("worktree", "/home/me/git/gigachad-bot-worktrees/feat", { project_root: "/home/me/git/gigachad-bot" }),
      session("sibling", "/home/me/git/gigachad-bot-old"),
    ]
    expect(sessionsInside(sessions, "/home/me/git/gigachad-bot/").map((found) => found.id)).toEqual(["sub", "worktree"])
  })
})

describe("sessionForest", () => {
  it("nests forks under their parent, newest first, and keeps orphaned forks as roots", () => {
    const forest = sessionForest([
      session("old", "/w", { modified: "2026-01-01" }),
      session("new", "/w", { modified: "2026-01-03" }),
      session("fork", "/w", { modified: "2026-01-02", parent_session_id: "old" }),
      session("orphan", "/w", { parent_session_id: "gone" }),
    ])
    expect(forest.map((node) => node.session.id)).toEqual(["new", "old", "orphan"])
    expect(forest[1].forks.map((node) => node.session.id)).toEqual(["fork"])
  })
})

describe("groupSessions", () => {
  const now = Date.parse("2026-06-10T12:00:00Z")
  const at = (ageMs: number) => new Date(now - ageMs).toISOString()
  const ids = (sessions: OmpSession[]) => sessions.map((found) => found.id)

  it("puts running sessions first whatever their age", () => {
    const groups = groupSessions([session("stale", "/w", { modified: at(10 * RECENT_SESSION_MS) })], ["stale"], now)
    expect(ids(groups.running)).toEqual(["stale"])
    expect(groups.recent).toEqual([])
    expect(groups.older).toEqual([])
  })

  it("keeps sessions modified within 24 hours recent and moves later ones to older", () => {
    const groups = groupSessions([
      session("edge", "/w", { modified: at(RECENT_SESSION_MS) }),
      session("past", "/w", { modified: at(RECENT_SESSION_MS + 1000) }),
    ], [], now)
    expect(ids(groups.recent)).toEqual(["edge"])
    expect(ids(groups.older)).toEqual(["past"])
  })

  it("orders each group newest first", () => {
    const groups = groupSessions([
      session("a", "/w", { modified: at(3000) }),
      session("b", "/w", { modified: at(1000) }),
      session("c", "/w", { modified: at(2000) }),
      session("r1", "/w", { modified: at(5000) }),
      session("r2", "/w", { modified: at(4000) }),
    ], ["r1", "r2"], now)
    expect(ids(groups.running)).toEqual(["r2", "r1"])
    expect(ids(groups.recent)).toEqual(["b", "c", "a"])
  })
})

describe("unlinkedWorkspaces", () => {
  it("groups sessions by their deepest workspace, leaves linked ones out and keeps empty registered workspaces", () => {
    const overview: OmpOverview = {
      running_session_ids: [],
      workspaces: [
        { path: "/git", alias: null, added_at: null },
        { path: "/git/tool", alias: "Tool", added_at: null },
        { path: "/git/linked/docs", alias: null, added_at: null },
        { path: "/empty", alias: null, added_at: null },
      ],
      sessions: [
        session("deep", "/git/tool/src"),
        session("shallow", "/git/misc"),
        session("linked", "/git/linked"),
        session("loose", "/tmp/scratch"),
      ],
    }
    const groups = unlinkedWorkspaces(overview, ["/git/linked"])
    expect(groups.map((group) => [group.label, group.sessions.map((found) => found.id)])).toEqual([
      ["empty", []],
      ["git", ["shallow"]],
      ["scratch", ["loose"]],
      ["Tool", ["deep"]],
    ])
  })
})

describe("startDirectories", () => {
  it("lists mounts and unmounted workspaces once, the most recently used first and unused ones by name", () => {
    const overview: OmpOverview = {
      running_session_ids: [],
      workspaces: [{ path: "/work/tool", alias: null, added_at: null }, { path: "/work/idle", alias: null, added_at: null }],
      sessions: [
        session("old", "/git/app", { modified: "2026-01-01" }),
        session("worktree", "/wt/feat", { project_root: "/git/lib", modified: "2026-03-01" }),
        session("tool", "/work/tool/src", { modified: "2026-02-01" }),
      ],
    }
    expect(startDirectories(["/git/app/", "/git/lib", "/git/zeta"], overview)).toEqual(["/git/lib", "/work/tool", "/git/app", "/work/idle", "/git/zeta"])
  })

  it("lists only the mounts while omp-web never answered", () => {
    expect(startDirectories(["/git/b", "/git/a"], null)).toEqual(["/git/a", "/git/b"])
  })
})
