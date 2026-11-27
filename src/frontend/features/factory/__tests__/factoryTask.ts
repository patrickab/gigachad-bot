import type { FactoryTask, FactoryWave, FactoryWaveTask } from "@/lib/domain/factory"

/** A core `TaskView` of a running task in `/git/repo`, `fields` over it. */
export function factoryTask(id: string, fields: Partial<FactoryTask> = {}): FactoryTask {
  return {
    id, repo: "/git/repo", worktree: `/worktrees/${id}`, branch: `factory/${id}`, title: id, tag: "me", plan: null, wave: null, question: null, createdAt: "2026-01-01T00:00:00Z",
    sessionId: `s-${id}`, sessionPath: `/sessions/s-${id}.jsonl`, state: "running", round: 1, reason: "", at: "2026-01-01T00:00:00Z",
    running: true, interrupted: false, actions: ["tasks.cancel", "tasks.message"], settings: null, usage: null, merged: null, ...fields,
  }
}

/** A declared task of a wave, not started yet, `fields` over it. */
export function waveTask(key: string, fields: Partial<FactoryWaveTask> = {}): FactoryWaveTask {
  return { key, title: key, after: [], taskId: null, sessionId: null, state: "pending", usage: null, ...fields }
}

/** A core `WaveView` of a building wave in `/git/repo` without tasks, `fields` over it. */
export function factoryWave(id: string, fields: Partial<FactoryWave> = {}): FactoryWave {
  return {
    id, repo: "/git/repo", tag: `p/${id}`, plan: null, title: id, request: "", branch: `master-p-${id}`, target: "master", base: "abc",
    origin: null, createdAt: "2026-01-01T00:00:00Z", stage: "building", reason: null, running: true, interrupted: false,
    tasks: [], counts: { total: 0, done: 0 }, reviews: [], gates: [], merge: null, remote: null, usage: null, actions: [], ...fields,
  }
}
