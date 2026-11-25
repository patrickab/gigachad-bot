import { describe, expect, it } from "vitest"
import { descriptionPath, groupByTag, type TagGroup } from "@/features/factory/tagGroups"
import { factoryTask } from "./factoryTask"
import type { FactoryTask } from "@/lib/domain/factory"

function task(id: string, tag: string, fields: Partial<FactoryTask> = {}): FactoryTask {
  return factoryTask(id, { tag, ...fields })
}

/** The tree as `path count [states] (task ids)`, depth first. */
function outline(groups: readonly TagGroup[]): string[] {
  return groups.flatMap((group) => [
    `${group.path} ${group.count} [${group.states.map(({ state, count }) => `${state}:${count}`).join(",")}] (${group.tasks.map((one) => one.id).join(",")})`,
    ...outline(group.children),
  ])
}

describe("groupByTag", () => {
  it("nests plan paths of depth 2 and 3 by segment and leaves single-segment tags for the columns", () => {
    const { groups, ungrouped } = groupByTag([
      task("a", "me"),
      task("b", "plugin/m1/w1"),
      task("c", "board/w1"),
      task("d", "plugin/m1/w2"),
      task("e", "noob"),
      task("f", "plugin/m2/w1"),
    ])

    expect(ungrouped.map((one) => one.id)).toEqual(["a", "e"])
    expect(outline(groups)).toEqual([
      "board 1 [] ()",
      "board/w1 1 [running:1] (c)",
      "plugin 3 [] ()",
      "plugin/m1 2 [] ()",
      "plugin/m1/w1 1 [running:1] (b)",
      "plugin/m1/w2 1 [running:1] (d)",
      "plugin/m2 1 [] ()",
      "plugin/m2/w1 1 [running:1] (f)",
    ])
  })

  it("counts a wave's tasks per shown state, in board order", () => {
    const { groups } = groupByTag([
      task("a", "p/w1", { state: "done", running: false }),
      task("b", "p/w1", { state: "waiting_answer", running: false }),
      task("c", "p/w1", { state: "done", running: false }),
      task("d", "p/w1", { state: "running", interrupted: true, running: false }),
      task("e", "p/w1", { state: null }),
    ])

    expect(groups[0].children[0].states).toEqual([
      { state: "waiting_answer", count: 1 }, { state: "interrupted", count: 1 }, { state: "starting", count: 1 }, { state: "done", count: 2 },
    ])
  })

  it("orders groups by name, numbers by value, and a wave's tasks newest first, whatever the listed order", () => {
    const tasks = [
      task("old", "p/w10", { createdAt: "2026-01-01T00:00:00Z" }),
      task("new", "p/w10", { createdAt: "2026-01-02T00:00:00Z" }),
      task("tie2", "p/w10", { createdAt: "2026-01-01T00:00:00Z" }),
      task("x", "p/w2"),
    ]
    const ordered = outline(groupByTag(tasks).groups)

    expect(ordered).toEqual(["p 4 [] ()", "p/w2 1 [running:1] (x)", "p/w10 3 [running:3] (new,old,tie2)"])
    expect(outline(groupByTag([...tasks].reverse()).groups)).toEqual(ordered)
  })

  it("names the plan of the first task in a group that has one", () => {
    const { groups } = groupByTag([task("a", "p/w1"), task("b", "p/w2", { plan: "gigachad-bot/factory-protocol" })])

    expect(groups[0].plan).toBe("gigachad-bot/factory-protocol")
    expect(groups[0].children.map((child) => child.plan)).toEqual([null, "gigachad-bot/factory-protocol"])
  })

  it("leaves a task of an older plugin, without tag or plan, in the columns and its siblings' plan intact", () => {
    const older: Partial<FactoryTask> = factoryTask("old")
    delete older.tag
    delete older.plan
    const planless = { ...task("a", "p/w1"), plan: undefined } as unknown as FactoryTask
    const { groups, ungrouped } = groupByTag([older as FactoryTask, planless, task("b", "p/w1", { plan: "store" })])

    expect(ungrouped.map((one) => one.id)).toEqual(["old"])
    expect(groups[0].plan).toBe("store")
  })

  it("finds a group's description under plan/ in the memory_store", () => {
    expect(descriptionPath("plugin/m1")).toBe("plan/plugin/m1.md")
  })
})
