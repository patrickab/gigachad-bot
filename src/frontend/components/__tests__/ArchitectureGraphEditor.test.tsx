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

  it("still autosaves while typing continues past the max wait", async () => {
    const source = await openSourceTab()

    for (let i = 0; i < Math.ceil(AUTOSAVE_MAX_WAIT_MS / 200) + 2; i += 1) {
      fireEvent.change(source, { target: { value: `version: 1\ntitle: T${i}\nnodes: []\nedges: []\n` } })
      await advance(200)
    }

    expect(api.writeArchitectureGraph).toHaveBeenCalled()
  })

  it("does not rewrite content that matches the loaded file", async () => {
    const source = await openSourceTab()

    fireEvent.change(source, { target: { value: VALID } })
    await advance(AUTOSAVE_MAX_WAIT_MS + 100)

    expect(api.writeArchitectureGraph).not.toHaveBeenCalled()
  })
})
