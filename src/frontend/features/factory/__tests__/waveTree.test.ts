import { describe, expect, it } from "vitest"
import { descriptionPath, descriptionTitle, treeOfWaves, waveColumn, type WaveGroup } from "@/features/factory/waveTree"
import type { FactoryWaveStage } from "@/lib/domain/factory"
import { factoryWave } from "./factoryTask"

/** The tree as `path merged/total cost (wave ids)`, depth first. */
function outline(group: WaveGroup): string[] {
  return group.children.flatMap((child) => [
    `${child.path} ${child.counts.merged}/${child.counts.total} ${child.usage?.cost ?? "-"} (${child.waves.map((wave) => wave.id).join(",")})`,
    ...outline(child),
  ])
}

describe("treeOfWaves", () => {
  it("builds tracks and milestones from the waves' tag prefixes, each counting its waves, the merged ones and their usage", () => {
    const root = treeOfWaves([
      factoryWave("a", { tag: "plugin/m1/w1", stage: "merged", usage: { input: 10, output: 5, cost: 1 } }),
      factoryWave("b", { tag: "plugin/m1/w2", usage: { input: 20, output: 5, cost: 2 } }),
      factoryWave("c", { tag: "plugin/m2/w1", stage: "merged" }),
      factoryWave("d", { tag: "board/w1", usage: { input: 1, output: 1, cost: 0.5 } }),
      factoryWave("e", { tag: "solo" }),
    ])

    expect(root.waves.map((wave) => wave.id)).toEqual(["e"])
    expect(outline(root)).toEqual([
      "board 0/1 0.5 (d)",
      "plugin 2/3 3 ()",
      "plugin/m1 1/2 3 (a,b)",
      "plugin/m2 1/1 - (c)",
    ])
    expect(root.children[1].usage).toEqual({ input: 30, output: 10, cost: 3 })
  })

  it("carries cache reads and writes through the sum", () => {
    const root = treeOfWaves([
      factoryWave("a", { tag: "p/w1", usage: { input: 10, output: 5, cost: 1, cacheRead: 100, cacheWrite: 7 } }),
      factoryWave("b", { tag: "p/w2", usage: { input: 20, output: 5, cost: 2, cacheRead: 50 } }),
      factoryWave("c", { tag: "p/w3", usage: { input: 1, output: 1, cost: 0 } }),
    ])
    expect(root.children[0].usage).toEqual({ input: 31, output: 11, cost: 3, cacheRead: 150, cacheWrite: 7 })
  })

  it("orders by name, numbers by value, whatever the listed order", () => {
    const waves = [factoryWave("ten", { tag: "p/w10" }), factoryWave("two", { tag: "p/w2" })]
    expect(treeOfWaves(waves).children[0].waves.map((wave) => wave.id)).toEqual(["two", "ten"])
    expect(treeOfWaves([...waves].reverse()).children[0].waves.map((wave) => wave.id)).toEqual(["two", "ten"])
  })

  it("names the plan of the first wave in a group that has one", () => {
    const root = treeOfWaves([factoryWave("a", { tag: "p/w1" }), factoryWave("b", { tag: "p/w2", plan: "store" })])
    expect(root.children[0].plan).toBe("store")
  })

  it("finds a description under plan/ in the memory_store", () => {
    expect(descriptionPath("plugin/m1")).toBe("plan/plugin/m1.md")
  })
})

describe("waveColumn", () => {
  const column = (stage: FactoryWaveStage, hasDone = false, interrupted = false) => waveColumn({ stage, interrupted }, hasDone)

  it("puts a wave being built, reviewed or merged in Merging, a building one only once a task is done", () => {
    expect(column("building")).toBeNull()
    expect(column("building", true)).toBe("merging")
    expect(column("reviewing")).toBe("merging")
    expect(column("merging")).toBe("merging")
  })

  it("puts a wave resting on the supervisor or interrupted in Waiting, a merged or cancelled one in Finished", () => {
    expect(["rejected", "blocked", "waiting_approve"].map((stage) => column(stage as FactoryWaveStage))).toEqual(["waiting", "waiting", "waiting"])
    expect(column("building", false, true)).toBe("waiting")
    expect(column("merged")).toBe("finished")
    expect(column("cancelled")).toBe("finished")
  })
})

describe("descriptionTitle", () => {
  it("is the first non-empty line without heading marks", () => {
    expect(descriptionTitle("\n## Plugin track\n\nBody")).toBe("Plugin track")
    expect(descriptionTitle("Plain first line\nmore")).toBe("Plain first line")
    expect(descriptionTitle(" \n#\n")).toBeNull()
  })
})
