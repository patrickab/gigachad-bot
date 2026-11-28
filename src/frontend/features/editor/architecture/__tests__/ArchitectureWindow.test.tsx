import { useState } from "react"
import { act, fireEvent, render, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { ArchitectureWorkspace } from "@/lib/domain/architecture"
import type { ChangeEvent } from "@/lib/backend/syncStream"

// The surface and the text editor have their own tests; here they only report what the window hands them.
vi.mock("@/features/editor/architecture/ArchitectureDiagramSurface", () => ({
  ArchitectureDiagramSurface: ({ select, onRemoveFromView }: { select?: string | null, onRemoveFromView?: (ids: string[]) => void }) => (
    <div data-testid="architecture-surface" data-select={select ?? ""}>
      <button type="button" onClick={() => onRemoveFromView?.(["backend.api"])}>remove-node</button>
    </div>
  ),
}))
vi.mock("@/features/editor/TextDocumentEditor", () => ({
  TextDocumentEditor: ({ path, persistOverride }: { path: string, persistOverride: (content: string) => Promise<void> }) => (
    <button type="button" data-testid="text-editor" onClick={() => { persistOverride("model {}").catch(() => {}) }}>{path}</button>
  ),
}))

const feed = vi.hoisted(() => new Set<(event: ChangeEvent) => void>())
vi.mock("@/lib/backend/syncStream", () => ({
  syncCursorReady: () => Promise.resolve(),
  subscribeToChanges: (listener: (event: ChangeEvent) => void) => {
    feed.add(listener)
    return () => { feed.delete(listener) }
  },
}))
/** Another device edited one of the project's architecture files. */
const remoteChange = () => act(() => feed.forEach((listener) => listener({ seq: 1, resource_kind: "document", resource_key: "graph/proj/backend/backend.c4", version: 2, operation: "write", device_id: "other-device" })))

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((settle) => { resolve = settle })
  return { promise, resolve }
}

const workspace: ArchitectureWorkspace = {
  created: [],
  model: {
    errors: [], kinds: ["system", "container"],
    elements: [
      { id: "backend", name: "backend", kind: "system", title: "Backend", description: "", parent: null, file: "backend/backend.c4" },
      { id: "backend.api", name: "api", kind: "container", title: "API", description: "", parent: "backend", file: "backend/api.c4" },
    ],
    views: [
      { id: "backend", title: "Backend", file: "backend/backend.c4", scope: "backend", editable: true, manual: true, nodes: [], edges: [] },
      { id: "checkout", title: "Checkout", file: "views/checkout.c4", scope: null, editable: true, manual: true, nodes: [], edges: [] },
    ],
    tree: [
      { path: "backend/api.c4", role: "module", element: "backend.api", view: "backend" },
      { path: "backend/backend.c4", role: "system", element: "backend", view: "backend" },
      { path: "views/checkout.c4", role: "view", element: null, view: "checkout" },
    ],
  },
}

/** `workspace` with its system retitled, to tell answers apart. */
const retitled = (title: string): ArchitectureWorkspace => ({
  ...workspace,
  model: { ...workspace.model, elements: workspace.model.elements.map((element) => element.id === "backend" ? { ...element, title } : element) },
})

const api = vi.hoisted(() => ({
  readArchitecture: vi.fn(),
  applyArchitectureOperations: vi.fn(),
  deleteArchitectureSources: vi.fn(),
  writeArchitectureSource: vi.fn(),
}))
vi.mock("@/lib/backend/api", () => api)

import { ArchitectureWindow } from "@/features/editor/architecture/ArchitectureWindow"
import { UndoDeleteProvider } from "@/contexts/UndoDeleteContext"
import { resetResources } from "@/lib/resource"
import { watchDocumentLists } from "@/lib/backend/resources"

function Harness() {
  const [path, setPath] = useState("graph/proj/")
  return <>
    <output data-testid="path">{path}</output>
    <ArchitectureWindow path={path} maximized={false} hostScale={1} onEdgeStyleChange={() => {}} onNavigate={setPath} />
  </>
}

/** Renders the window, which opens on the first system of `initial`. */
async function open(initial: ArchitectureWorkspace = workspace) {
  api.readArchitecture.mockResolvedValueOnce(initial)
  const view = render(<UndoDeleteProvider><Harness /></UndoDeleteProvider>)
  await act(async () => {})
  const button = (text: string) => Array.from(view.container.querySelectorAll("button")).find((b) => b.textContent === text)!
  /** Names a new system, which writes it. */
  const createSystem = async (title: string) => {
    act(() => { view.getByLabelText("New system").click() })
    const field = view.getByLabelText("System name")
    fireEvent.change(field, { target: { value: title } })
    await act(async () => { fireEvent.keyDown(field, { key: "Enter" }) })
  }
  return { ...view, button, createSystem }
}

describe("ArchitectureWindow", () => {
  let unwatch = () => {}
  beforeEach(() => {
    vi.resetAllMocks()
    resetResources()
    unwatch = watchDocumentLists()
  })
  afterEach(() => {
    unwatch()
    vi.useRealTimers()
  })

  it("opens on the first system and moves through its tree", async () => {
    const { button, createSystem, getByLabelText, getByRole, findByRole, getByTestId } = await open()
    api.deleteArchitectureSources.mockResolvedValue(workspace)
    const path = () => getByTestId("path").textContent

    expect(path()).toBe("graph/proj/backend/backend.c4")
    expect(getByTestId("architecture-surface").dataset.select).toBe("")

    // A module opens in its system's view, selected.
    act(() => { button("API").click() })
    await act(async () => {})
    expect(path()).toBe("graph/proj/backend/api.c4")
    expect(getByTestId("architecture-surface").dataset.select).toBe("backend.api")

    // A new module lands beside its system and opens.
    api.applyArchitectureOperations.mockResolvedValueOnce({ ...workspace, created: ["backend/db.c4"] })
    act(() => { getByLabelText("New module in Backend").click() })
    const moduleField = getByLabelText("Module name")
    fireEvent.change(moduleField, { target: { value: "DB" } })
    await act(async () => { fireEvent.keyDown(moduleField, { key: "Enter" }) })
    expect(api.applyArchitectureOperations).toHaveBeenLastCalledWith("proj", [{ op: "createModule", system: "backend", title: "DB" }], null)
    expect(path()).toBe("graph/proj/backend/db.c4")

    // A refused name keeps the field open with the server's reason.
    api.applyArchitectureOperations.mockRejectedValueOnce(new Error("Unknown element kind"))
    await createSystem("Frontend")
    expect((await findByRole("alert")).textContent).toContain("Unknown element kind")
    expect(getByLabelText("System name")).toBeTruthy()

    // The same file as text, then the system deleted together with its modules.
    act(() => { button("Backend").click() })
    await act(async () => {})
    act(() => { button("Text").click() })
    expect(getByTestId("text-editor").textContent).toBe("graph/proj/backend/backend.c4")
    vi.useFakeTimers()
    act(() => { getByRole("button", { name: "Delete Backend" }).click() })
    expect(api.deleteArchitectureSources).not.toHaveBeenCalled()
    expect(button("Undo")).toBeTruthy()
    // The system and its modules leave the tree at once.
    expect(button("Backend")).toBeUndefined()
    expect(button("API")).toBeUndefined()
    // Moving on during the undo window keeps the window where it now is once the delete lands.
    act(() => { button("Checkout").click() })
    await act(async () => { vi.advanceTimersByTime(6000) })
    expect(api.deleteArchitectureSources).toHaveBeenLastCalledWith("proj", ["backend/backend.c4", "backend/api.c4"])
    expect(path()).toBe("graph/proj/views/checkout.c4")
  })

  it("asks whether an element removed from a saved view should leave its file too", async () => {
    const { button, getByRole, queryByRole } = await open()
    act(() => { button("Checkout").click() })
    await act(async () => {})
    api.applyArchitectureOperations.mockResolvedValue(workspace)

    // Escape keeps it in the model.
    await act(async () => { button("remove-node").click() })
    expect(api.applyArchitectureOperations).toHaveBeenLastCalledWith("proj", [{ op: "removeFromView", view: "checkout", elements: ["backend.api"] }], "views/checkout.c4")
    expect(getByRole("alertdialog").textContent).toContain("Remove reference from backend/api.c4?")
    await act(async () => { fireEvent.keyDown(window, { key: "Escape" }) })
    expect(queryByRole("alertdialog")).toBeNull()
    expect(api.applyArchitectureOperations).toHaveBeenCalledTimes(1)

    // Enter deletes it from the model.
    await act(async () => { button("remove-node").click() })
    await act(async () => { fireEvent.keyDown(window, { key: "Enter" }) })
    expect(api.applyArchitectureOperations).toHaveBeenLastCalledWith("proj", [{ op: "delete", elements: ["backend.api"], relations: [] }], "views/checkout.c4")
    expect(queryByRole("alertdialog")).toBeNull()
  })

  it("shows every window of one project the same model, read once", async () => {
    api.readArchitecture.mockResolvedValue(workspace)
    const window = (onNavigate: (path: string) => void) =>
      <ArchitectureWindow path="graph/proj/" maximized={false} hostScale={1} onEdgeStyleChange={() => {}} onNavigate={onNavigate} />
    const { findAllByText, queryAllByText, getAllByLabelText, getByLabelText } = render(<UndoDeleteProvider>{window(() => {})}{window(() => {})}</UndoDeleteProvider>)
    await act(async () => {})
    expect(api.readArchitecture).toHaveBeenCalledTimes(1)

    // An edit in the first window reaches the second.
    api.applyArchitectureOperations.mockResolvedValueOnce(retitled("Edited"))
    act(() => { getAllByLabelText("New system")[0].click() })
    const field = getByLabelText("System name")
    fireEvent.change(field, { target: { value: "Frontend" } })
    await act(async () => { fireEvent.keyDown(field, { key: "Enter" }) })
    expect(await findAllByText("Edited")).toHaveLength(2)

    // Another device's change is read once for both.
    api.readArchitecture.mockResolvedValue(retitled("Changed elsewhere"))
    remoteChange()
    expect(await findAllByText("Changed elsewhere")).toHaveLength(2)
    expect(queryAllByText("Edited")).toHaveLength(0)
    expect(api.readArchitecture).toHaveBeenCalledTimes(2)
  })

  it("ends two windows' writes on the later write's answer", async () => {
    api.readArchitecture.mockResolvedValue(workspace)
    const window = <ArchitectureWindow path="graph/proj/" maximized={false} hostScale={1} onEdgeStyleChange={() => {}} onNavigate={() => {}} />
    const { findAllByText, getAllByLabelText } = render(<UndoDeleteProvider>{window}{window}</UndoDeleteProvider>)
    await act(async () => {})
    const first = deferred<ArchitectureWorkspace>()
    api.applyArchitectureOperations.mockReturnValueOnce(first.promise).mockResolvedValueOnce(retitled("Second"))
    for (const index of [0, 1]) {
      act(() => { getAllByLabelText("New system")[index].click() })
      fireEvent.change(getAllByLabelText("System name")[index], { target: { value: "Frontend" } })
      await act(async () => { fireEvent.keyDown(getAllByLabelText("System name")[index], { key: "Enter" }) })
    }
    await act(async () => { first.resolve(retitled("First")) })
    expect(await findAllByText("Second")).toHaveLength(2)
  })

  it("reads a model another device changed while no window showed it", async () => {
    const { unmount } = await open()
    unmount()
    remoteChange()
    const { findByText } = await open(retitled("Changed elsewhere"))
    await findByText("Changed elsewhere")
  })

  it("never lets an earlier read replace a later write's answer", async () => {
    const { createSystem, queryByText } = await open()
    const read = deferred<ArchitectureWorkspace>()
    const write = deferred<ArchitectureWorkspace>()
    api.readArchitecture.mockReturnValueOnce(read.promise)
    api.applyArchitectureOperations.mockReturnValueOnce(write.promise)

    remoteChange()
    await createSystem("Frontend")
    // The write's answer arrives first, the earlier read's after it, which is dropped and read again.
    api.readArchitecture.mockResolvedValueOnce(retitled("Written"))
    await act(async () => {
      write.resolve(retitled("Written"))
      read.resolve(retitled("Read"))
    })
    await waitFor(() => expect(api.readArchitecture).toHaveBeenCalledTimes(3))
    expect(queryByText("Written")).toBeTruthy()
    expect(queryByText("Read")).toBeNull()
  })

  it("rereads once after a write when another device changed the model meanwhile", async () => {
    const { createSystem, findByText, getByText } = await open()
    const write = deferred<ArchitectureWorkspace>()
    api.applyArchitectureOperations.mockReturnValueOnce(write.promise)
    api.readArchitecture.mockResolvedValue(retitled("Changed elsewhere"))

    await createSystem("Frontend")
    remoteChange()
    remoteChange()
    await findByText("Changed elsewhere")
    expect(api.readArchitecture).toHaveBeenCalledTimes(2)

    // The write's answer predates the other device's change, so the model is read once more.
    await act(async () => { write.resolve(workspace) })
    await waitFor(() => expect(api.readArchitecture).toHaveBeenCalledTimes(3))
    expect(getByText("Changed elsewhere")).toBeTruthy()
  })

  it("shows a failed reread until a later one succeeds", async () => {
    const { findByRole, queryByRole } = await open()

    api.readArchitecture.mockRejectedValueOnce(new Error("Storage unavailable"))
    remoteChange()
    expect((await findByRole("alert")).textContent).toBe("Storage unavailable")

    api.readArchitecture.mockResolvedValueOnce(workspace)
    remoteChange()
    await waitFor(() => expect(queryByRole("alert")).toBeNull())
  })

  it("shows a text save's answer without reading the model again", async () => {
    const { button, getByTestId, findByText } = await open()
    act(() => { button("Text").click() })
    api.writeArchitectureSource.mockResolvedValueOnce(retitled("Saved as text"))
    act(() => { getByTestId("text-editor").click() })
    await act(async () => {})
    act(() => { button("Diagram").click() })

    expect(api.writeArchitectureSource).toHaveBeenCalledWith("proj", "backend/backend.c4", "model {}")
    await findByText("Saved as text")
    expect(api.readArchitecture).toHaveBeenCalledTimes(1)
  })

  it("lists LikeC4's complaints with 1-based lines, even with no view to draw", async () => {
    const { getByText } = await open({
      ...workspace,
      model: { ...workspace.model, views: [], errors: [{ file: "backend/backend.c4", line: 2, message: "Could not resolve 'db'" }] },
    })
    expect(getByText("backend/backend.c4 declares no view yet.")).toBeTruthy()
    expect(getByText("backend/backend.c4:3: Could not resolve 'db'")).toBeTruthy()
  })
})
