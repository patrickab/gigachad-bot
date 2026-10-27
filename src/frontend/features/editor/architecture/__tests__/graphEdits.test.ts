import { describe, expect, it } from "vitest"
import { isDraft, type DiagramNode } from "@/features/editor/architecture/diagramGraph"
import { duplicateNodes } from "@/features/editor/architecture/graphEdits"

const node = (id: string, title: string, position = { x: 0, y: 0 }, extra: Partial<DiagramNode> = {}): DiagramNode =>
  ({ id, title, bullets: [], kind: "system", parent: null, compound: false, position, width: 240, height: 160, ...extra })

describe("duplicateNodes", () => {
  it("clones nodes as titled drafts at an offset position, with ids free of the graph and of each other", () => {
    const nodes = [node("api", "A", { x: 100, y: 100 }), node("db", "B", { x: 300, y: 100 })]
    const clones = duplicateNodes(nodes, ["api", "db", "~draft-1"])
    // A draft id makes the copy a new element rather than a second view of `api`.
    expect(clones.map((clone) => isDraft(clone.id))).toEqual([true, true])
    expect(new Set([...clones.map((clone) => clone.id), "~draft-1"]).size).toBe(3)
    expect(clones[0].title).toBe("A")
    expect(clones[0].position).toEqual({ x: 128, y: 128 })
  })

  it("skips compounds, whose children would not come along", () => {
    expect(duplicateNodes([node("shop", "Shop", { x: 0, y: 0 }, { compound: true })], ["shop"])).toEqual([])
  })
})
