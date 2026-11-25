import type { FactoryTask } from "@/lib/domain/factory"

/** A core `TaskView` of a running task in `/git/repo`, `fields` over it. */
export function factoryTask(id: string, fields: Partial<FactoryTask> = {}): FactoryTask {
  return {
    id, repo: "/git/repo", worktree: `/worktrees/${id}`, branch: `factory/${id}`, title: id, tag: "me", plan: null, kind: "task", question: null, createdAt: "2026-01-01T00:00:00Z",
    sessionId: `s-${id}`, sessionPath: `/sessions/s-${id}.jsonl`, state: "running", round: 1, reason: "", at: "2026-01-01T00:00:00Z",
    running: true, interrupted: false, actions: ["tasks.cancel", "tasks.message"], settings: null, usage: null, ...fields,
  }
}
