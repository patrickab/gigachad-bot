import { describe, expect, it } from "vitest"
import { graphToMermaidFlowchart, parseMermaidFlowchart } from "@/lib/mermaidGraph"
import { emptyArchitectureGraph, type ArchitectureGraph } from "@/lib/architectureGraph"

describe("parseMermaidFlowchart", () => {
  it("parses shapes, directions, labels, and explicit edge ids", () => {
    const source = `
      flowchart LR
      %% a comment line is ignored
      api[Checkout API]
      db((Database))
      gate{Approved?}
      api e1@--> db
      db e2@<-->|sync| gate
    `.split("\n").map((line) => line.trim()).filter(Boolean).join("\n")

    const graph = parseMermaidFlowchart(source, "Checkout")
    expect(graph.nodes.map((n) => [n.id, n.title, n.shape])).toEqual([
      ["api", "Checkout API", undefined],
      ["db", "Database", "ellipse"],
      ["gate", "Approved?", "diamond"],
    ])
    expect(graph.edges).toEqual([
      { id: "e1", source: "api", target: "db", direction: "one-way" },
      { id: "e2", source: "db", target: "gate", direction: "bidirectional", label: "sync" },
    ])
  })

  it("implicitly declares nodes referenced only inside an edge line", () => {
    const graph = parseMermaidFlowchart("flowchart TB\nA --> B")
    expect(graph.nodes.map((n) => n.id)).toEqual(["A", "B"])
    expect(graph.nodes[0].title).toBe("A")
  })

  it("splits a <br/> label into a title and bullets", () => {
    const graph = parseMermaidFlowchart("flowchart TB\napi[API<br/>• validates carts]")
    expect(graph.nodes[0]).toMatchObject({ title: "API", bullets: ["validates carts"] })
  })

  it("rejects a missing or unsupported header", () => {
    expect(() => parseMermaidFlowchart("A --> B")).toThrow("flowchart")
    expect(() => parseMermaidFlowchart("sequenceDiagram\nA->>B: hi")).toThrow("flowchart")
  })

  it("rejects a duplicate explicit edge id", () => {
    const source = "flowchart TB\nA e1@--> B\nB e1@--> C"
    expect(() => parseMermaidFlowchart(source)).toThrow("Duplicate edge id")
  })

  it("round-trips a node label containing arrow, pipe, and bracket characters", () => {
    const nodes = [{ id: "n1", title: "Queue --> Worker | array[0]", bullets: [], position: { x: 0, y: 0 } }]
    const edges = [{ id: "e1", source: "n1", target: "n1", direction: "one-way" as const, label: "yes | no" }]
    const graph: ArchitectureGraph = { ...emptyArchitectureGraph("Checkout"), nodes, edges }
    const reparsed = parseMermaidFlowchart(graphToMermaidFlowchart(graph))
    expect(reparsed.nodes[0].title).toBe(nodes[0].title)
    expect(reparsed.edges[0].label).toBe("yes | no")
  })

  it("only treats '%%' as a comment at the start of a line", () => {
    const graph = parseMermaidFlowchart("flowchart TB\nn1[50%% off]")
    expect(graph.nodes[0].title).toBe("50%% off")
  })
})

describe("graphToMermaidFlowchart", () => {
  it("round-trips ids, shapes, labels, and bullets through parsing", () => {
    const nodes = [
      { id: "api", title: "API", bullets: ["validates carts"], position: { x: 0, y: 0 } },
      { id: "db", title: "Database", bullets: [], position: { x: 0, y: 0 }, shape: "ellipse" as const },
      { id: "gate", title: "Approved?", bullets: [], position: { x: 0, y: 0 }, shape: "diamond" as const },
    ]
    const edges = [
      { id: "e1", source: "api", target: "db", direction: "one-way" as const },
      { id: "e2", source: "db", target: "gate", direction: "bidirectional" as const, label: "sync" },
    ]
    const graph: ArchitectureGraph = { ...emptyArchitectureGraph("Checkout"), nodes, edges }

    const mermaid = graphToMermaidFlowchart(graph)
    expect(mermaid).toContain("flowchart TB")
    expect(mermaid).toContain("db e2@<-->")
    const reparsed = parseMermaidFlowchart(mermaid, "Checkout")
    expect(reparsed.nodes.map((n) => [n.id, n.title, n.bullets, n.shape])).toEqual(
      graph.nodes.map((n) => [n.id, n.title, n.bullets, n.shape]),
    )
    expect(reparsed.edges).toEqual(graph.edges)
  })
})
