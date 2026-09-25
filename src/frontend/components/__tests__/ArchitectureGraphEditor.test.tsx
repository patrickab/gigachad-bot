import type { ComponentType, ReactNode } from "react"
import { act, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { ArchitectureGraphEditor } from "@/components/ArchitectureGraphEditor"
import { AUTOSAVE_DEBOUNCE_MS, AUTOSAVE_MAX_WAIT_MS } from "@/lib/graphAutosave"

const VALID = "version: 1\ntitle: Test graph\nnodes: []\nedges: []\n"

const api = vi.hoisted(() => ({
  readArchitectureGraph: vi.fn(),
  writeArchitectureGraph: vi.fn(async () => ({})),
}))

vi.mock("@/lib/api", () => api)

vi.mock("@xyflow/react", async () => {
  const { useState } = await import("react")
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
    NodeResizer: () => null,
    Handle: () => null,
    getBezierPath: () => ["", 0, 0],
    useInternalNode: () => undefined,
    useNodesState: useItemsState,
    useEdgesState: useItemsState,
    MarkerType: { ArrowClosed: "arrowclosed" },
    ConnectionMode: { Strict: "strict", Loose: "loose" },
    Position: { Top: "top", Left: "left", Right: "right", Bottom: "bottom" },
  }
})

async function openSourceTab(name = "demo.architecture.yaml") {
  api.readArchitectureGraph.mockResolvedValue({ name, path: `/graphs/${name}`, content: VALID, hasDraft: false })
  render(<ArchitectureGraphEditor path={`/graphs/${name}`} onClose={vi.fn()} />)
  await act(async () => { await Promise.resolve() })
  fireEvent.click(screen.getByRole("tab", { name: "Source" }))
  return screen.getByRole("textbox", { name: "Architecture Graph YAML source" })
}

const advance = async (ms: number) => {
  await act(async () => { await vi.advanceTimersByTimeAsync(ms) })
}

describe("ArchitectureGraphEditor autosave", () => {
  beforeEach(() => {
    vi.useFakeTimers()
    api.writeArchitectureGraph.mockClear()
  })
  afterEach(() => vi.useRealTimers())

  it("never writes a Source buffer that does not parse", async () => {
    const source = await openSourceTab()

    fireEvent.change(source, { target: { value: "version: 1\ntitle: [broken" } })
    await advance(AUTOSAVE_MAX_WAIT_MS * 2)

    expect(api.writeArchitectureGraph).not.toHaveBeenCalled()
    expect(screen.getByRole("alert")).toBeInTheDocument()
    // The broken text stays editable rather than snapping back.
    expect(source).toHaveValue("version: 1\ntitle: [broken")
  })

  it("autosaves a valid Source edit once the debounce settles", async () => {
    const source = await openSourceTab()
    const edited = "version: 1\ntitle: Renamed\nnodes: []\nedges: []\n"

    fireEvent.change(source, { target: { value: edited } })
    await advance(AUTOSAVE_DEBOUNCE_MS + 10)

    expect(api.writeArchitectureGraph).toHaveBeenCalledTimes(1)
    expect(api.writeArchitectureGraph).toHaveBeenCalledWith("demo.architecture.yaml", edited)
  })

})

describe("ArchitectureGraphEditor undo/redo", () => {
  it("undoes and redoes a Diagram-view edit with Ctrl+Z / Ctrl+Shift+Z", async () => {
    api.readArchitectureGraph.mockResolvedValue({ name: "demo.architecture.yaml", path: "/graphs/demo.architecture.yaml", content: VALID, hasDraft: false })
    render(<ArchitectureGraphEditor path="/graphs/demo.architecture.yaml" onClose={vi.fn()} />)
    await act(async () => { await Promise.resolve() })

    fireEvent.click(screen.getByRole("button", { name: "Add node" }))
    // A new node opens empty, straight into title editing.
    expect(screen.getByRole("textbox", { name: "Node title" })).toHaveFocus()
    expect(screen.getByRole("textbox", { name: "Node title" })).toHaveValue("")

    fireEvent.keyDown(document, { key: "z", ctrlKey: true })
    expect(screen.queryByRole("textbox", { name: "Node title" })).not.toBeInTheDocument()

    // Redo restores the node without re-entering edit mode: the auto-edit is one-time.
    fireEvent.keyDown(document, { key: "z", ctrlKey: true, shiftKey: true })
    expect(screen.getByRole("button", { name: "Edit node title" })).toHaveTextContent("Untitled node")
    expect(screen.queryByRole("textbox", { name: "Node title" })).not.toBeInTheDocument()
  })
})
