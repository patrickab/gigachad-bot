import { describe, expect, it } from "vitest"
import { isInside, projectMount, sessionForest, sessionsInside } from "@/features/omp/sessions"
import type { OmpSession } from "@/lib/domain/types"

function session(id: string, cwd: string, extra: Partial<OmpSession> = {}): OmpSession {
  return {
    id, path: `/s/${id}.jsonl`, cwd, name: null, created: "2026-01-01", modified: "2026-01-01", message_count: 1,
    first_message: id, parent_session_id: null, project_root: null, worktree_branch: null, ...extra,
  }
}

describe("isInside", () => {
  it("matches the directory and below it, never a sibling sharing its prefix", () => {
    expect(isInside("/a/foo", "/a/foo")).toBe(true)
    expect(isInside("/a/foo/src", "/a/foo/")).toBe(true)
    expect(isInside("/a/foobar", "/a/foo")).toBe(false)
    expect(isInside("/a", "/a/foo")).toBe(false)
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
