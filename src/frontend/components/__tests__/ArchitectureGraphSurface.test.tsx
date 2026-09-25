import type { ComponentType, ReactNode } from "react"
import { act, fireEvent, render, screen } from "@testing-library/react"
import { useEffect, useState } from "react"
import { describe, expect, it, vi } from "vitest"
import { ArchitectureGraphSurface, classifyDrawnShape, duplicateNodes, edgeAttachments, snapToGrid } from "@/components/ArchitectureGraphSurface"
import { emptyArchitectureGraph, type ArchitectureGraph } from "@/lib/architectureGraph"

const flow = vi.hoisted(() => ({
  fitView: vi.fn(),
  zoomIn: vi.fn(),
  zoomOut: vi.fn(),
}))

vi.mock("@xyflow/react", () => {
  // Mirrors React Flow's own useNodesState/useEdgesState: plain state plus a change handler.
  const useItemsState = <T,>(initial: T) => {
    const [items, setItems] = useState(initial)
    return [items, setItems, vi.fn()] as const
  }
  return {
    ReactFlow: ({ children, nodes = [], nodeTypes = {}, onInit }: { children: ReactNode; nodes?: Array<{ id: string; type?: string; data: Record<string, unknown> }>; nodeTypes?: Record<string, ComponentType<{ data: Record<string, unknown>; selected: boolean }>>; onInit?: (instance: typeof flow) => void }) => {
      useEffect(() => { onInit?.(flow) }, [onInit])
      return <div data-testid="flow">{nodes.map((node) => {
        const NodeType = node.type ? nodeTypes[node.type] : undefined
        return NodeType ? <NodeType key={node.id} data={node.data} selected={false} /> : null
      })}{children}</div>
    },
    Background: () => null,
    BaseEdge: () => null,
    NodeResizer: () => null,
    Handle: () => null,
    getSmoothStepPath: () => ["", 0, 0],
    useInternalNode: () => undefined,
    useNodesState: useItemsState,
    useEdgesState: useItemsState,
    MarkerType: { ArrowClosed: "arrowclosed" },
    ConnectionMode: { Strict: "strict", Loose: "loose" },
    Position: { Top: "top", Left: "left", Right: "right", Bottom: "bottom" },
  }
})

describe("ArchitectureGraphSurface", () => {
  it("gives connections sharing a card side their own slot on it", () => {
    const nodes = [
      { id: "left", position: { x: 0, y: 0 }, measured: { width: 100, height: 60 } },
      { id: "right", position: { x: 300, y: 0 }, measured: { width: 100, height: 60 } },
      { id: "below", position: { x: 300, y: 300 }, measured: { width: 100, height: 60 } },
    ]
    const edge = (id: string, source: string, target: string) => ({ id, source, target, direction: "one-way" as const })

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

  it("attaches connections to the drawn outline of round shapes, not their bounding box", () => {
    const edge = (id: string, source: string, target: string) => ({ id, source, target, direction: "one-way" as const })
    const nodes = [
      { id: "diamond", position: { x: 0, y: 0 }, measured: { width: 100, height: 100 }, data: { shape: "diamond" as const } },
      { id: "ellipse", position: { x: 400, y: 0 }, measured: { width: 200, height: 100 }, data: { shape: "ellipse" as const } },
      { id: "below", position: { x: 0, y: 300 }, measured: { width: 100, height: 60 } },
    ]
    // Three connections share the diamond's bottom side: off-center slots must
    // land on its slanted edges, not in the empty box corner below them.
    const attachments = edgeAttachments(nodes, [edge("a", "diamond", "below"), edge("b", "diamond", "below"), edge("c", "diamond", "below"), edge("d", "diamond", "ellipse")])
    for (const id of ["a", "b", "c"]) {
      const { x, y } = attachments.get(id)!.source
      expect(Math.abs(x - 50) / 50 + Math.abs(y - 50) / 50).toBeCloseTo(1)
    }
    expect(attachments.get("a")!.source.y).toBeLessThan(100)
    const { x, y } = attachments.get("d")!.target
    expect(((x - 500) / 100) ** 2 + ((y - 50) / 50) ** 2).toBeCloseTo(1)
  })

  it("keeps the title editor focused while controlled graph updates arrive", () => {
    const initialGraph = { ...emptyArchitectureGraph(), nodes: [{ id: "node-1", title: "Gateway", bullets: [], position: { x: 0, y: 0 } }] }
    function ControlledSurface() {
      const [graph, setGraph] = useState<ArchitectureGraph>(initialGraph)
      return <ArchitectureGraphSurface graph={graph} onChange={setGraph} />
    }

    render(<ControlledSurface />)
    fireEvent.click(screen.getByRole("button", { name: "Edit node title" }))
    const title = screen.getByRole("textbox", { name: "Node title" })
    fireEvent.change(title, { target: { value: "Gateway API" } })

    expect(title).toHaveFocus()
    expect(title).toHaveValue("Gateway API")
  })

  it("commits an in-progress title to the graph without disturbing the caret", async () => {
    vi.useFakeTimers()
    try {
      const initialGraph = { ...emptyArchitectureGraph(), nodes: [{ id: "node-1", title: "Gateway", bullets: [], position: { x: 0, y: 0 } }] }
      const seen = vi.fn()
      function ControlledSurface() {
        const [graph, setGraph] = useState<ArchitectureGraph>(initialGraph)
        return <ArchitectureGraphSurface graph={graph} onChange={(next) => { seen(next); setGraph(next) }} />
      }

      render(<ControlledSurface />)
      fireEvent.click(screen.getByRole("button", { name: "Edit node title" }))
      const title = screen.getByRole("textbox", { name: "Node title" }) as HTMLInputElement
      fireEvent.change(title, { target: { value: "Gateway API" } })
      title.setSelectionRange(7, 7)

      // Autosave depends on typed text reaching the graph before any blur.
      await act(async () => { await vi.advanceTimersByTimeAsync(400) })

      expect(seen.mock.calls.at(-1)?.[0].nodes[0].title).toBe("Gateway API")
      expect(title).toHaveFocus()
      expect(title).toHaveValue("Gateway API")
      expect(title.selectionStart).toBe(7)
    } finally {
      vi.useRealTimers()
    }
  })

  it("zooms from the viewport center on wheel", () => {
    flow.zoomIn.mockClear()
    flow.zoomOut.mockClear()

    render(<ArchitectureGraphSurface graph={emptyArchitectureGraph()} onChange={() => {}} />)
    fireEvent.wheel(screen.getByTestId("flow"), { deltaY: -100 })

    expect(flow.zoomIn).toHaveBeenCalledOnce()
    expect(flow.zoomOut).not.toHaveBeenCalled()
  })
})

describe("classifyDrawnShape", () => {
  const perimeter = (points: Array<[number, number]>, steps = 8) => {
    const out: Array<{ x: number, y: number }> = []
    for (let i = 0; i < points.length; i += 1) {
      const [ax, ay] = points[i]
      const [bx, by] = points[(i + 1) % points.length]
      for (let s = 0; s < steps; s += 1) out.push({ x: ax + (bx - ax) * (s / steps), y: ay + (by - ay) * (s / steps) })
    }
    return out
  }

  it("recognizes a rough rectangle", () => {
    const points = perimeter([[0, 0], [100, 0], [100, 60], [0, 60]])
    const result = classifyDrawnShape(points)
    expect(result?.shape).toBe("rectangle")
    expect(result?.box).toEqual({ x: 0, y: 0, width: 100, height: 60 })
  })

  it("recognizes a diamond", () => {
    const points = perimeter([[50, 0], [100, 30], [50, 60], [0, 30]])
    expect(classifyDrawnShape(points)?.shape).toBe("diamond")
  })

  it("recognizes an ellipse", () => {
    const points = Array.from({ length: 40 }, (_, i) => {
      const angle = (i / 40) * Math.PI * 2
      return { x: 50 + Math.cos(angle) * 50, y: 30 + Math.sin(angle) * 30 }
    })
    expect(classifyDrawnShape(points)?.shape).toBe("ellipse")
  })

  it("rejects an open stroke", () => {
    const points = Array.from({ length: 10 }, (_, i) => ({ x: i * 10, y: 0 }))
    expect(classifyDrawnShape(points)).toBeNull()
  })

  it("rejects a stroke smaller than the minimum draw size", () => {
    const points = perimeter([[0, 0], [10, 0], [10, 10], [0, 10]])
    expect(classifyDrawnShape(points)).toBeNull()
  })
})

describe("snapToGrid", () => {
  it("rounds to the nearest 8px grid step", () => {
    expect(snapToGrid(13)).toBe(16)
    expect(snapToGrid(11)).toBe(8)
    expect(snapToGrid(0)).toBe(0)
  })
})

describe("duplicateNodes", () => {
  it("clones only the selected nodes with fresh ids and an offset position", () => {
    const nodes = [
      { id: "node-1", title: "A", bullets: [], position: { x: 100, y: 100 } },
      { id: "node-2", title: "B", bullets: [], position: { x: 300, y: 100 } },
    ]
    const clones = duplicateNodes(nodes, new Set(["node-1"]))
    expect(clones).toHaveLength(1)
    expect(clones[0].id).not.toBe("node-1")
    expect(clones[0].title).toBe("A")
    expect(clones[0].position).toEqual({ x: 128, y: 128 })
  })

  it("returns no clones when nothing is selected", () => {
    const nodes = [{ id: "node-1", title: "A", bullets: [], position: { x: 0, y: 0 } }]
    expect(duplicateNodes(nodes, new Set())).toEqual([])
  })
})
