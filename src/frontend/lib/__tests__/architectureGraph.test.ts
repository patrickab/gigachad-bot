import { describe, expect, it } from "vitest"
import { emptyArchitectureGraph, parseArchitectureGraph, serializeArchitectureGraph } from "@/lib/architectureGraph"

describe("Architecture Graph YAML", () => {
  it("round-trips the portable graph shape", () => {
    const nodes = [
      { id: "api", title: "API", bullets: ["validates carts"], position: { x: 10, y: 20 } },
      { id: "db", title: "Database", bullets: [], position: { x: 300, y: 120 } },
    ]
    const edges = [{ id: "writes", source: "api", target: "db", direction: "one-way" as const, path: { bend: 40 } }]
    const graph = { ...emptyArchitectureGraph("Checkout"), nodes, edges }
    expect(parseArchitectureGraph(serializeArchitectureGraph(graph))).toEqual(graph)
  })

  it("rejects dangling connections", () => {
    expect(() => parseArchitectureGraph("version: 1\ntitle: Test\nnodes: []\nedges:\n  - id: e\n    source: missing\n    target: missing\n    direction: one-way\n")).toThrow("missing node")
  })

  it("rejects a malformed connection bend", () => {
    const source = "version: 1\ntitle: Test\nnodes:\n  - id: a\n    title: A\n    bullets: []\n    position: { x: 0, y: 0 }\n  - id: b\n    title: B\n    bullets: []\n    position: { x: 10, y: 10 }\nedges:\n  - id: e\n    source: a\n    target: b\n    direction: one-way\n    path: { bend: nope }\n"
    expect(() => parseArchitectureGraph(source)).toThrow("path.bend must be a finite number")
  })

  it("drops legacy size metadata so height follows content", () => {
    const graph = parseArchitectureGraph("version: 1\ntitle: Test\nnodes:\n  - id: api\n    title: API\n    bullets: []\n    position: { x: 0, y: 0 }\n    size: { width: 320, height: 180 }\nedges: []\n")
    expect(graph.nodes[0]).not.toHaveProperty("size")
  })
})
