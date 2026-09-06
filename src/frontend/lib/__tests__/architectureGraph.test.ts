import { describe, expect, it } from "vitest"
import { emptyArchitectureGraph, nextArchitectureGraphId, parseArchitectureGraph, serializeArchitectureGraph } from "@/lib/architectureGraph"

describe("Architecture Graph YAML", () => {
  it("round-trips the portable graph shape", () => {
    const graph = { ...emptyArchitectureGraph("Checkout"), nodes: [{ id: "api", title: "API", bullets: ["validates carts"], position: { x: 10, y: 20 } }], edges: [] }
    expect(parseArchitectureGraph(serializeArchitectureGraph(graph))).toEqual(graph)
  })

  it("rejects dangling connections", () => {
    expect(() => parseArchitectureGraph("version: 1\ntitle: Test\nnodes: []\nedges:\n  - id: e\n    source: missing\n    target: missing\n    direction: one-way\n")).toThrow("missing node")
  })

  it("creates stable concise identifiers", () => {
    expect(nextArchitectureGraphId("node", ["node-1", "node-2", "node-4"])).toBe("node-3")
  })

  it("drops legacy size metadata so height follows content", () => {
    const graph = parseArchitectureGraph("version: 1\ntitle: Test\nnodes:\n  - id: api\n    title: API\n    bullets: []\n    position: { x: 0, y: 0 }\n    size: { width: 320, height: 180 }\nedges: []\n")
    expect(graph.nodes[0]).not.toHaveProperty("size")
  })
})
