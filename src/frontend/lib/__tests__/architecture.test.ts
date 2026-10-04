import { describe, expect, it } from "vitest"
import { diffGraph, fitGraph, type DiagramEdge, type DiagramGraph, type DiagramNode } from "@/lib/architecture"

const KINDS = ["actor", "component", "container", "database", "system"]

const node = (id: string, extra: Partial<DiagramNode> = {}): DiagramNode =>
  ({ id, title: id, bullets: [], kind: "system", parent: null, compound: false, position: { x: 0, y: 0 }, width: 240, height: 160, ...extra })
const edge = (source: string, target: string, relations = [`${source}-${target}`], extra: Partial<DiagramEdge> = {}): DiagramEdge =>
  ({ id: `${source}->${target}`, source, target, relations, ...extra })

const base: DiagramGraph = {
  nodes: [node("shop", { compound: true, width: 600, height: 400 }), node("shop.api", { parent: "shop", kind: "container", position: { x: 40, y: 60 } }), node("bank", { position: { x: 800, y: 0 } })],
  edges: [edge("shop.api", "bank", ["r1"])],
}

describe("diffGraph", () => {
  it("deletes only what was removed: a node keeps its connections unless they were removed too", () => {
    const nodeOnly = diffGraph(base, { ...base, nodes: base.nodes.filter((n) => n.id !== "bank") }, "index", KINDS)
    expect(nodeOnly.ops).toEqual([{ op: "delete", elements: ["bank"], relations: [] }])

    const both = diffGraph(base, { nodes: base.nodes.filter((n) => n.id !== "bank"), edges: [] }, "index", KINDS)
    expect(both.ops).toEqual([{ op: "delete", elements: ["bank"], relations: ["r1"] }])
  })

  it("keeps an untitled drawn node local, then creates it once it has a title", () => {
    const drawn = node("~draft-1", { title: "", kind: "", parent: "shop", position: { x: 100, y: 200 } })
    const untitled = diffGraph(base, { ...base, nodes: [...base.nodes, drawn] }, "index", KINDS)
    expect(untitled.ops).toEqual([])
    expect(untitled.drafts).toEqual([drawn])

    const named = diffGraph({ ...base, nodes: [...base.nodes, drawn] }, { ...base, nodes: [...base.nodes, { ...drawn, title: "Ledger", bullets: ["books", "audits"] }] }, "index", KINDS)
    expect(named.drafts).toEqual([])
    // A drawn node is a component unless a kind was chosen.
    expect(named.ops).toEqual([{
      op: "addElement", parent: "shop", kind: "component", title: "Ledger", description: "books\naudits",
      layout: { view: "index", x: 100, y: 200, width: 240, height: 160 },
    }])
  })

  it("moves a node as a layout change only, never a model edit", () => {
    const moved = diffGraph(base, { ...base, nodes: base.nodes.map((n) => n.id === "bank" ? { ...n, position: { x: 808, y: 16 } } : n) }, "index", KINDS)
    expect(moved.ops).toEqual([{ op: "layout", view: "index", nodes: { bank: { x: 808, y: 16, width: 240, height: 160 } }, edges: {} }])
  })

  it("reparents a node dropped into another compound and saves its position under the new FQN", () => {
    const dropped = diffGraph(base, { ...base, nodes: base.nodes.map((n) => n.id === "bank" ? { ...n, parent: "shop", position: { x: 300, y: 80 } } : n) }, "index", KINDS)
    expect(dropped.ops).toEqual([
      { op: "reparent", element: "bank", parent: "shop" },
      { op: "layout", view: "index", nodes: { "shop.bank": { x: 300, y: 80, width: 240, height: 160 } }, edges: {} },
    ])
  })

  it("relabels a single relationship but refuses a connection that merges several", () => {
    const single = diffGraph(base, { ...base, edges: [{ ...base.edges[0], label: "pays" }] }, "index", KINDS)
    expect(single.ops).toEqual([{ op: "setLabel", relation: "r1", label: "pays" }])

    const merged = { ...base, edges: [edge("shop.api", "bank", ["r1", "r2"])] }
    const refused = diffGraph(merged, { ...merged, edges: [{ ...merged.edges[0], label: "pays" }] }, "index", KINDS)
    expect(refused.ops).toEqual([])
    expect(refused.rejected).toMatch(/merges 2 relationships/)
  })

  it("adds a drawn connection, but not one to a node that has no title yet", () => {
    const added = diffGraph(base, { ...base, edges: [...base.edges, { ...edge("bank", "shop", []), id: "~draft-1" }] }, "index", KINDS)
    expect(added.ops).toEqual([{ op: "addRelation", source: "bank", target: "shop" }])

    const drawn = node("~draft-1", { title: "" })
    const before = { ...base, nodes: [...base.nodes, drawn] }
    const refused = diffGraph(before, { ...before, edges: [...base.edges, { ...edge("bank", "~draft-1", []), id: "~draft-2" }] }, "index", KINDS)
    expect(refused.ops).toEqual([])
    expect(refused.drafts).toEqual([drawn])
    expect(refused.rejected).toBeTruthy()
  })

  it("changes a node's kind in place", () => {
    const changed = diffGraph(base, { ...base, nodes: base.nodes.map((n) => n.id === "shop.api" ? { ...n, kind: "database" } : n) }, "index", KINDS)
    expect(changed.ops).toEqual([{ op: "setKind", element: "shop.api", kind: "database" }])
  })
})

describe("fitGraph", () => {
  it("widens a card to its label, pushes the sibling it now overlaps, and grows every parent up the tree", () => {
    const nodes = [
      node("svc", { compound: true, width: 300, height: 200 }),
      node("svc.a", { parent: "svc", position: { x: 32, y: 64 }, width: 100, height: 60 }),
      node("svc.b", { parent: "svc", position: { x: 140, y: 64 }, width: 100, height: 60 }),
      node("bank", { position: { x: 320, y: 0 }, width: 100, height: 60 }),
      node("far", { position: { x: 0, y: 900 }, width: 100, height: 60 }),
    ]
    const fitted = new Map(fitGraph(nodes, (n) => (n.id === "svc.a" ? 160 : 80)).map((n) => [n.id, n]))

    expect(fitted.get("svc.a")!.width).toBe(160)
    expect(fitted.get("svc.b")!.position.x).toBe(32 + 160 + 24)
    expect(fitted.get("svc")!.width).toBe(216 + 100 + 32)
    expect(fitted.get("bank")!.position.x).toBe(348 + 24)
    expect(fitted.get("far")).toBe(nodes[4])
    // Already fitting: nothing moves.
    const again = fitGraph([...fitted.values()], (n) => (n.id === "svc.a" ? 160 : 80))
    expect(again).toEqual([...fitted.values()])
  })
})
