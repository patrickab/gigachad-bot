import type { ComponentType, ReactNode } from "react"
import { act, fireEvent, render, screen } from "@testing-library/react"
import { useState } from "react"
import { describe, expect, it, vi } from "vitest"
import { ArchitectureGraphSurface, edgeAttachments } from "@/components/ArchitectureGraphSurface"
import { emptyArchitectureGraph, type ArchitectureGraph } from "@/lib/architectureGraph"

vi.mock("@xyflow/react", async () => {
  const { useState } = await import("react")
  // Mirrors React Flow's own useNodesState/useEdgesState: plain state plus a change handler.
  const useItemsState = <T,>(initial: T) => {
    const [items, setItems] = useState(initial)
    return [items, setItems, vi.fn()] as const
  }
  return {
    ReactFlow: ({ children, nodes = [], nodeTypes = {} }: { children: ReactNode; nodes?: Array<{ id: string; type?: string; data: Record<string, unknown> }>; nodeTypes?: Record<string, ComponentType<{ data: Record<string, unknown>; selected: boolean }>> }) => <div data-testid="flow">{nodes.map((node) => {
      const NodeType = node.type ? nodeTypes[node.type] : undefined
      return NodeType ? <NodeType key={node.id} data={node.data} selected={false} /> : null
    })}{children}</div>,
    Background: () => null,
    BaseEdge: () => null,
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
})
