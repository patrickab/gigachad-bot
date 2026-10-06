import type { ComponentType, ReactNode } from "react"
import { act, fireEvent, render, screen } from "@testing-library/react"
import { useEffect, useState } from "react"
import { describe, expect, it, vi } from "vitest"
import { ArchitectureDiagramSurface, drawnBox, duplicateNodes, edgeAttachments, type ArchitectureDiagramSurfaceProps } from "@/components/ArchitectureDiagramSurface"
import { isDraft, type DiagramGraph, type DiagramNode } from "@/lib/architecture"

const node = (id: string, title: string, position = { x: 0, y: 0 }, extra: Partial<DiagramNode> = {}): DiagramNode =>
  ({ id, title, bullets: [], kind: "system", parent: null, compound: false, position, ...extra })
const emptyGraph: DiagramGraph = { nodes: [], edges: [] }
const Surface = (props: Omit<ArchitectureDiagramSurfaceProps, "edgeStyle" | "onEdgeStyleChange"> & Partial<ArchitectureDiagramSurfaceProps>) =>
  <ArchitectureDiagramSurface edgeStyle="elbow" onEdgeStyleChange={() => {}} {...props} />

const flow = vi.hoisted(() => ({
  fitView: vi.fn(),
  screenToFlowPosition: (point: { x: number, y: number }) => point,
}))
// The surface's Delete-key handler, as React Flow would call it.
const deleteKey = vi.hoisted(() => ({ onBeforeDelete: null as null | ((doomed: { nodes: Array<{ id: string, selected: boolean }>, edges: Array<{ id: string, selected: boolean }> }) => Promise<boolean>) }))

vi.mock("@xyflow/react", () => {
  // Mirrors React Flow's own useNodesState/useEdgesState: plain state plus a change handler.
  const useItemsState = <T,>(initial: T) => {
    const [items, setItems] = useState(initial)
    return [items, setItems, vi.fn()] as const
  }
  return {
    ReactFlow: ({ children, nodes = [], nodeTypes = {}, onInit, onBeforeDelete }: { children: ReactNode; nodes?: Array<{ id: string; type?: string; selected?: boolean; data: Record<string, unknown> }>; nodeTypes?: Record<string, ComponentType<{ data: Record<string, unknown>; selected: boolean }>>; onInit?: (instance: typeof flow) => void; onBeforeDelete?: typeof deleteKey.onBeforeDelete }) => {
      deleteKey.onBeforeDelete = onBeforeDelete ?? null
      useEffect(() => { onInit?.(flow) }, [onInit])
      return <div data-testid="flow" className="react-flow__pane">{nodes.map((node) => {
        const NodeType = node.type ? nodeTypes[node.type] : undefined
        return NodeType ? <div key={node.id} className={node.selected ? "react-flow__node selected" : "react-flow__node"}><NodeType data={node.data} selected={!!node.selected} /></div> : null
      })}{children}</div>
    },
    Background: () => null,
    BaseEdge: () => null,
    NodeResizer: () => null,
    Handle: () => null,
    useInternalNode: () => undefined,
    useNodesState: useItemsState,
    useEdgesState: useItemsState,
    ConnectionMode: { Loose: "loose" },
    Position: { Top: "top", Left: "left", Right: "right", Bottom: "bottom" },
  }
})
// No WebAssembly here: connections stay unrouted, as before libavoid loads.
vi.mock("@/lib/edgeRouting", () => ({ loadRouter: () => Promise.resolve(false), routeEdges: () => null, bendPoint: vi.fn(), bendThrough: vi.fn() }))

describe("ArchitectureDiagramSurface", () => {
  it("draws with the pen into a card that is not activated, and in a saved view nowhere else", async () => {
    const onChange = vi.fn()
    render(<Surface graph={{ nodes: [node("bank", "Bank")], edges: [] }} onChange={onChange} topLevelDrawing={false} />)
    const loop = (start: Element, x: number, y: number) => {
      // jsdom has no PointerEvent, so the pen is a mouse event that says it is a pen.
      const down = new MouseEvent("pointerdown", { bubbles: true, button: 0, clientX: x, clientY: y })
      Object.defineProperty(down, "pointerType", { value: "pen" })
      start.dispatchEvent(down)
      for (const [dx, dy] of [[180, 0], [180, 70], [0, 70], [2, 2]]) window.dispatchEvent(new MouseEvent("pointermove", { clientX: x + dx!, clientY: y + dy! }))
      window.dispatchEvent(new MouseEvent("pointerup"))
    }

    loop(screen.getByTestId("flow"), 600, 600)
    expect(onChange).not.toHaveBeenCalled()

    loop(screen.getByLabelText("Edit node title").closest(".architecture-diagram-node")!, 10, 10)
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ nodes: [expect.objectContaining({ id: "bank" }), expect.objectContaining({ parent: "bank" })] }))
    // Lets the stroke's click guard expire, as it does a tick after the pen lifts.
    await act(() => new Promise((resolve) => setTimeout(resolve)))
  })

  it("in a saved view, the Delete key and the corner X take nodes out of the view but connections still leave the model", async () => {
    const onChange = vi.fn()
    const onRemoveFromView = vi.fn()
    const graph: DiagramGraph = { nodes: [node("shop", "Shop"), node("bank", "Bank")], edges: [{ id: "shop->bank", source: "shop", target: "bank", relations: ["r1"] }] }
    render(<Surface graph={graph} onChange={onChange} onRemoveFromView={onRemoveFromView} />)

    await act(() => deleteKey.onBeforeDelete!({ nodes: [{ id: "bank", selected: true }], edges: [{ id: "shop->bank", selected: true }] }))
    expect(onRemoveFromView).toHaveBeenCalledWith(["bank"])
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ nodes: graph.nodes, edges: [] }))

    fireEvent.mouseEnter(screen.getAllByLabelText("Edit node title")[0]!.closest(".architecture-diagram-node")!)
    fireEvent.click(screen.getByLabelText("Remove from view"))
    expect(onRemoveFromView).toHaveBeenLastCalledWith(["shop"])
    expect(screen.queryByLabelText("Delete element")).toBeNull()
  })

  it("in a system view, the corner X deletes the element from the model", () => {
    const onChange = vi.fn()
    const graph: DiagramGraph = { nodes: [node("shop", "Shop"), node("bank", "Bank")], edges: [] }
    render(<Surface graph={graph} onChange={onChange} />)

    fireEvent.mouseEnter(screen.getAllByLabelText("Edit node title")[1]!.closest(".architecture-diagram-node")!)
    fireEvent.click(screen.getByLabelText("Delete element"))
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ nodes: [graph.nodes[0]] }))
  })

  it("changes an activated node's kind from its kind label", () => {
    const onChange = vi.fn()
    const graph: DiagramGraph = { nodes: [node("shop", "Shop"), node("shop.db", "DB", { x: 0, y: 0 }, { kind: "container", parent: "shop" })], edges: [] }
    render(<Surface graph={graph} onChange={onChange} select="shop.db" kinds={["component", "container", "database", "system"]} />)

    const label = screen.getAllByLabelText("Kind")[1]!
    expect(label.textContent).toBe("container")
    fireEvent.click(label)
    fireEvent.click(screen.getByRole("option", { name: "database" }))
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ nodes: [graph.nodes[0], { ...graph.nodes[1], kind: "database" }] }))
  })

  it("writes a heading, then bullets, and commits them together once Enter on an empty bullet leaves the card", () => {
    const onChange = vi.fn()
    render(<Surface graph={{ nodes: [node("~draft-1", "")], edges: [] }} onChange={onChange} />)
    const committed = () => onChange.mock.calls.some(([graph]) => (graph as DiagramGraph).nodes.some((candidate) => candidate.title))

    fireEvent.click(screen.getByLabelText("Edit node title"))
    const title = screen.getByLabelText("Node title")
    fireEvent.change(title, { target: { value: "Ledger" } })
    fireEvent.keyDown(title, { key: "Enter" })
    const first = screen.getByLabelText("Node bullet")
    expect(document.activeElement).toBe(first)
    fireEvent.change(first, { target: { value: "books" } })
    fireEvent.keyDown(first, { key: "Enter" })
    const second = screen.getAllByLabelText("Node bullet")[1]!
    expect(document.activeElement).toBe(second)
    // Still inside the card: a new node must not become an element mid-edit.
    expect(committed()).toBe(false)

    act(() => { fireEvent.keyDown(second, { key: "Enter" }) })
    expect(document.activeElement).not.toBe(second)
    expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ nodes: [expect.objectContaining({ id: "~draft-1", title: "Ledger", bullets: ["books"] })] }))
  })

  it("starts a new bullet when the space below the bullets is clicked, and drops it again if left empty", () => {
    render(<Surface graph={{ nodes: [node("shop", "Shop", { x: 0, y: 0 }, { bullets: ["sells"] })], edges: [] }} onChange={() => {}} />)
    fireEvent.click(screen.getByLabelText("Node bullet").closest(".architecture-diagram-node-body")!)
    const bullets = screen.getAllByLabelText("Node bullet") as HTMLTextAreaElement[]
    expect(bullets.map((bullet) => bullet.value)).toEqual(["sells", ""])
    expect(document.activeElement).toBe(bullets[1])

    act(() => { bullets[1]!.blur() })
    expect((screen.getAllByLabelText("Node bullet") as HTMLTextAreaElement[]).map((bullet) => bullet.value)).toEqual(["sells"])
  })

  it("switches connections to curved from the overflow menu, dropping every saved bend", () => {
    const onChange = vi.fn()
    const onEdgeStyleChange = vi.fn()
    const graph: DiagramGraph = { nodes: [node("shop", "Shop"), node("bank", "Bank")], edges: [{ id: "shop->bank", source: "shop", target: "bank", relations: ["r1"], path: { bend: 12 } }] }
    render(<Surface graph={graph} onChange={onChange} onEdgeStyleChange={onEdgeStyleChange} />)

    fireEvent.click(screen.getByRole("button", { name: "Connection style" }))
    expect(screen.getByRole("menuitemradio", { name: "Angled edges" })).toHaveAttribute("aria-checked", "true")

    fireEvent.click(screen.getByRole("menuitemradio", { name: "Curved edges" }))
    expect(onEdgeStyleChange).toHaveBeenCalledWith("curved")
    expect(onChange).toHaveBeenCalledWith({ ...graph, edges: [{ id: "shop->bank", source: "shop", target: "bank", relations: ["r1"] }] })
  })

  it("gives connections sharing a card side their own slot on it", () => {
    const nodes = [
      { id: "left", position: { x: 0, y: 0 }, measured: { width: 100, height: 60 } },
      { id: "right", position: { x: 300, y: 0 }, measured: { width: 100, height: 60 } },
      { id: "below", position: { x: 300, y: 300 }, measured: { width: 100, height: 60 } },
    ]
    const edge = (id: string, source: string, target: string) => ({ id, source, target, relations: [] })

    // A lone connection still leaves from the side midpoint.
    const single = edgeAttachments(nodes, [edge("a", "left", "right")])
    expect(single.get("a")).toEqual({
      source: { x: 100, y: 30, position: "right" },
      target: { x: 300, y: 30, position: "left" },
    })

    // Two connections between the same pair share a side, so they split it.
    const pair = edgeAttachments(nodes, [edge("a", "left", "right"), edge("b", "left", "right")])
    expect(pair.get("a")!.source.y).not.toBe(pair.get("b")!.source.y)
    for (const attachment of pair.values()) {
      expect(attachment.source.x).toBe(100)
      expect(attachment.source.y).toBeGreaterThan(0)
      expect(attachment.source.y).toBeLessThan(60)
    }

    // Connections leaving different sides keep the midpoint of their own side.
    const fanned = edgeAttachments(nodes, [edge("a", "left", "right"), edge("b", "left", "below")])
    expect(fanned.get("a")!.source).toEqual({ x: 100, y: 30, position: "right" })
    expect(fanned.get("b")!.source).toEqual({ x: 50, y: 60, position: "bottom" })
  })

  it("keeps the title being edited when the graph is updated from outside", () => {
    const { rerender } = render(<Surface graph={{ nodes: [node("gateway", "Gateway")], edges: [] }} onChange={() => {}} />)
    fireEvent.click(screen.getByRole("button", { name: "Edit node title" }))
    const title = screen.getByRole("textbox", { name: "Node title" })
    fireEvent.change(title, { target: { value: "Gateway API" } })

    rerender(<Surface graph={{ nodes: [node("gateway", "Edge gateway"), node("bank", "Bank")], edges: [] }} onChange={() => {}} />)
    expect(title).toHaveFocus()
    expect(title).toHaveValue("Gateway API")
  })

  it("pastes the copied node as it was when copied, as a new draft", () => {
    const onChange = vi.fn()
    const original = node("api", "API", { x: 96, y: 96 }, { bullets: ["serves"], width: 240, height: 160 })
    const { rerender } = render(<Surface graph={{ nodes: [original], edges: [] }} onChange={onChange} select="api" />)
    fireEvent.keyDown(window, { key: "c", ctrlKey: true })

    const edited = { ...original, title: "Edited", bullets: [], position: { x: 400, y: 400 }, width: 320 }
    rerender(<Surface graph={{ nodes: [edited], edges: [] }} onChange={onChange} select="api" />)
    fireEvent.keyDown(window, { key: "v", ctrlKey: true })

    const [kept, pasted] = (onChange.mock.lastCall![0] as DiagramGraph).nodes
    expect(kept).toEqual(edited)
    const { id, ...copy } = pasted!
    expect(isDraft(id)).toBe(true)
    expect(copy).toEqual({ title: "API", bullets: ["serves"], kind: "system", parent: null, compound: false, position: { x: 120, y: 120 }, width: 240, height: 160 })
  })

  it("commits a title once, when editing ends, not on every keystroke", async () => {
    vi.useFakeTimers()
    try {
      const seen = vi.fn()
      function ControlledSurface() {
        const [graph, setGraph] = useState<DiagramGraph>({ nodes: [node("gateway", "Gateway")], edges: [] })
        return <Surface graph={graph} onChange={(next) => { seen(next); setGraph(next) }} />
      }

      render(<ControlledSurface />)
      fireEvent.click(screen.getByRole("button", { name: "Edit node title" }))
      const title = screen.getByRole("textbox", { name: "Node title" })
      fireEvent.change(title, { target: { value: "Gateway API" } })
      // Every graph change is a model edit, so a pause in typing must not send one.
      await act(async () => { await vi.advanceTimersByTimeAsync(1000) })
      expect(seen.mock.calls.some(([graph]) => graph.nodes[0].title === "Gateway API")).toBe(false)

      fireEvent.blur(title)
      expect(seen.mock.calls.at(-1)?.[0].nodes[0].title).toBe("Gateway API")
    } finally {
      vi.useRealTimers()
    }
  })

  it("refits the graph when its host maximizes it, but not when it restores", async () => {
    flow.fitView.mockClear()
    const nextFrames = () => act(async () => {
      const { promise, resolve } = Promise.withResolvers<void>()
      requestAnimationFrame(() => requestAnimationFrame(() => resolve()))
      await promise
    })
    const { rerender } = render(<Surface graph={emptyGraph} onChange={() => {}} />)
    await nextFrames()
    expect(flow.fitView).not.toHaveBeenCalled()

    rerender(<Surface graph={emptyGraph} onChange={() => {}} autoFit />)
    await nextFrames()
    expect(flow.fitView).toHaveBeenCalledOnce()

    rerender(<Surface graph={emptyGraph} onChange={() => {}} autoFit={false} />)
    await nextFrames()
    expect(flow.fitView).toHaveBeenCalledOnce()
  })
})

describe("drawnBox", () => {
  const perimeter = (points: Array<[number, number]>, steps = 8) => {
    const out: Array<{ x: number, y: number }> = []
    for (let i = 0; i < points.length; i += 1) {
      const [ax, ay] = points[i]
      const [bx, by] = points[(i + 1) % points.length]
      for (let s = 0; s < steps; s += 1) out.push({ x: ax + (bx - ax) * (s / steps), y: ay + (by - ay) * (s / steps) })
    }
    return out
  }

  it("turns any closed loop into its bounding box", () => {
    expect(drawnBox(perimeter([[0, 0], [100, 0], [100, 60], [0, 60]]))).toEqual({ x: 0, y: 0, width: 100, height: 60 })
    expect(drawnBox(perimeter([[50, 0], [100, 30], [50, 60], [0, 30]]))).toEqual({ x: 0, y: 0, width: 100, height: 60 })
  })

  it("rejects an open stroke", () => {
    const points = Array.from({ length: 10 }, (_, i) => ({ x: i * 10, y: 0 }))
    expect(drawnBox(points)).toBeNull()
  })

  it("rejects a stroke smaller than the minimum draw size", () => {
    expect(drawnBox(perimeter([[0, 0], [10, 0], [10, 10], [0, 10]]))).toBeNull()
  })
})

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
