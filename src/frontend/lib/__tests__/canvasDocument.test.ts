import { describe, expect, it } from "vitest"
import { diffEntities, historyChanges, emptyCanvasDoc } from "@/lib/canvasDocument"
import { strokesInLasso } from "@/lib/canvasSelection"
import type { StrokeData } from "@/lib/drawing"

const item = (id: string, value = 0) => ({ id, value })

describe("diffEntities", () => {
  it("emits upserts and new entities in `after` order, then deletions in `before` order", () => {
    const before = [item("a"), item("b"), item("c"), item("d")]
    const after = [item("n1"), item("a"), item("c", 1), item("n2")]
    expect(diffEntities(before, after).map((change) => [change.id, change.before?.id ?? null, change.after?.id ?? null])).toEqual([
      ["n1", null, "n1"],
      ["c", "c", "c"],
      ["n2", null, "n2"],
      ["b", "b", null],
      ["d", "d", null],
    ])
  })

  it("skips unchanged references and entities the comparator calls equal", () => {
    const a = item("a")
    expect(diffEntities([a, item("b", 1)], [a, item("b", 2)], (x, y) => x.id === y.id)).toEqual([])
  })
})

describe("historyChanges", () => {
  it("ignores an attachment's page, which is navigation rather than an edit", () => {
    const attachment = { id: "pdf", kind: "pdf" as const, path: "a.pdf", x: 0, y: 0, width: 500 }
    const before = { ...emptyCanvasDoc(), attachments: [attachment] }
    const after = { ...emptyCanvasDoc(), attachments: [{ ...attachment, page: 3 }] }
    expect(historyChanges(before, after)).toEqual([])
  })
})

describe("strokesInLasso", () => {
  const stroke = (id: string, x: number): StrokeData => ({ id, points: [[x, 5], [x + 1, 5]], color: "#000", width: 2 })
  const loop: [number, number][] = [[0, 0], [20, 0], [20, 10], [0, 10]]

  it("selects strokes by id, mostly-covered strokes only", () => {
    expect([...strokesInLasso([stroke("in", 5), stroke("out", 50), stroke("half", 19.5)], loop)]).toEqual(["in"])
  })
})
