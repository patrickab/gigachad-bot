import type { ComponentType, ReactNode } from "react"
import { act, fireEvent, render, screen } from "@testing-library/react"
import { useState } from "react"
import { describe, expect, it, vi } from "vitest"
import { ArchitectureGraphSurface } from "@/components/ArchitectureGraphSurface"
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
  it("uses a dedicated graph toolbar instead of React Flow's default controls", () => {
    render(<ArchitectureGraphSurface graph={emptyArchitectureGraph()} onChange={vi.fn()} />)

    expect(screen.getByRole("button", { name: "Add node" })).toBeVisible()
    expect(screen.getByRole("button", { name: "Fit view" })).toBeVisible()
    expect(screen.queryByText("Drag cards · scroll to zoom")).not.toBeInTheDocument()
  })

  it("reserves the card header for dragging except when its title is edited", () => {
    const graph = { ...emptyArchitectureGraph(), nodes: [{ id: "node-1", title: "Gateway", bullets: [], position: { x: 0, y: 0 } }] }
    render(<ArchitectureGraphSurface graph={graph} onChange={vi.fn()} />)

    expect(screen.getByRole("button", { name: "Edit node title" })).toBeVisible()
    fireEvent.click(screen.getByRole("button", { name: "Edit node title" }))
    expect(screen.getByRole("textbox", { name: "Node title" })).toBeVisible()
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
