import { describe, expect, it } from "vitest"
import { descriptionPath, treeOfWaves, type WaveGroup } from "@/features/factory/waveTree"
import { factoryWave } from "./factoryTask"

/** The tree as `path done/total cost (wave ids)`, depth first. */
function outline(group: WaveGroup): string[] {
  return group.children.flatMap((child) => [
    `${child.path} ${child.counts.done}/${child.counts.total} ${child.usage?.cost ?? "-"} (${child.waves.map((wave) => wave.id).join(",")})`,
    ...outline(child),
  ])
}

describe("treeOfWaves", () => {
  it("builds tracks and milestones from the waves' tag prefixes, each summing its waves", () => {
    const root = treeOfWaves([
      factoryWave("a", { tag: "plugin/m1/w1", counts: { total: 3, done: 1 }, usage: { input: 10, output: 5, cost: 1 } }),
      factoryWave("b", { tag: "plugin/m1/w2", counts: { total: 2, done: 2 }, usage: { input: 20, output: 5, cost: 2 } }),
      factoryWave("c", { tag: "plugin/m2/w1", counts: { total: 4, done: 0 } }),
      factoryWave("d", { tag: "board/w1", counts: { total: 1, done: 1 }, usage: { input: 1, output: 1, cost: 0.5 } }),
      factoryWave("e", { tag: "solo" }),
    ])

    expect(root.waves.map((wave) => wave.id)).toEqual(["e"])
    expect(outline(root)).toEqual([
      "board 1/1 0.5 (d)",
      "plugin 3/9 3 ()",
      "plugin/m1 3/5 3 (a,b)",
      "plugin/m2 0/4 - (c)",
    ])
    expect(root.children[1].usage).toEqual({ input: 30, output: 10, cost: 3 })
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
