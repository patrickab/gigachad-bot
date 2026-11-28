import { describe, expect, it } from "vitest"
import { outlineOfWaves } from "@/features/pi/waveOutline"
import type { FactoryWave, FactoryWaveTask } from "@/lib/domain/factory"

function waveTask(key: string, fields: Partial<FactoryWaveTask> = {}): FactoryWaveTask {
  return { key, title: key, after: [], taskId: null, sessionId: null, state: "pending", usage: null, ...fields }
}

function factoryWave(id: string, fields: Partial<FactoryWave> = {}): FactoryWave {
  return {
    id, repo: "/git/repo", tag: `p/${id}`, plan: null, title: id, request: "", branch: `master-p-${id}`, target: "master", base: "abc",
    origin: null, createdAt: "2026-01-01T00:00:00Z", stage: "building", reason: null, running: true, interrupted: false,
    tasks: [], counts: { total: 0, done: 0 }, reviews: [], gates: [], merge: null, remote: null, usage: null, actions: [], ...fields,
  }
}

const NOW = Date.parse("2026-01-01T00:08:30Z")
const labels = (waves: Parameters<typeof outlineOfWaves>[0], toggled: string[] = []) => outlineOfWaves(waves, new Set(toggled), NOW, {}).rows.map((row) => `${row.lead}${row.label}`)

const WAVES = [
  factoryWave("a", { tag: "plugin/m1/w1", stage: "merged", running: false, usage: { input: 1, output: 1, cost: 2 } }),
  factoryWave("b", {
    tag: "plugin/m1/w2", usage: { input: 1, output: 1, cost: 1.5 }, counts: { total: 3, done: 1 },
    tasks: [waveTask("rules", { state: "done", taskId: "t1" }), waveTask("readers", { state: "running", taskId: "t2" }), waveTask("cut", { after: ["readers"] })],
  }),
  factoryWave("c", { tag: "plugin/m2/w1", stage: "queued", running: false, after: ["b"], tasks: [waveTask("x")] }),
  factoryWave("d", { tag: "docs/w1", stage: "reviewing" }),
  factoryWave("e", { tag: "docs/w2", stage: "rejected", running: false }),
]

describe("outlineOfWaves", () => {
  it("nests the waves in their tags, joins a group of one into its wave, puts what needs the user first and folds what is finished", () => {
    expect(labels(WAVES)).toEqual([
      "├─ docs",
      "│  ├─ w2",
      "│  └─ w1",
      "└─ plugin",
      "   ├─ m1",
      "   │  ├─ w2",
      "   │  │  ├─ rules",
      "   │  │  ├─ readers",
      "   │  │  └─ cut",
      "   │  └─ 1 finished",
      "   └─ m2/w1",
    ])
  })

  it("opens and closes the toggled rows", () => {
    expect(labels(WAVES, ["group:docs", "wave:b", "wave:c", "finished:plugin/m1/"])).toEqual([
      "├─ docs",
      "└─ plugin",
      "   ├─ m1",
      "   │  ├─ w2",
      "   │  └─ 1 finished",
      "   │     └─ w1",
      "   └─ m2/w1",
      "      └─ x",
    ])
  })

  it("sums the spend, counts the coders and reviewers at work and the active waves, and gives every row its facts", () => {
    const { summary, rows } = outlineOfWaves(WAVES, new Set(), NOW, { t2: "opus" })
    expect(summary.map((item) => item.text)).toEqual(["$3.50", "2 workers", "4 active"])
    const facts = (label: string) => rows.find((row) => row.label === label)!.facts.map((item) => item.text)
    expect(facts("m1")).toEqual(["1/2 waves", "1 worker", "$3.50"])
    expect(facts("w2")).toEqual(["0/0 tasks", "rejected", "8m"])
    expect(rows.find((row) => row.key === "wave:b")!.facts.map((item) => item.text)).toEqual(["1/3 tasks", "building", "1 worker", "8m", "$1.50"])
    expect(facts("m2/w1")).toEqual(["0/0 tasks", "queued", "after plugin/m1/w2"])
    expect(facts("cut")).toEqual(["pending", "after readers"])
    expect(rows.find((row) => row.label === "readers")).toMatchObject({ task: { repo: "/git/repo", taskId: "t2" }, mark: "working" })
    expect(facts("readers")).toEqual(["opus", "running"])
  })
})
