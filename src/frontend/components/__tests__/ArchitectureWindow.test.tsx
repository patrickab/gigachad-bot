import { useState } from "react"
import { act, fireEvent, render, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { ArchitectureWorkspace } from "@/lib/architecture"
import type { ChangeEvent } from "@/lib/syncStream"

// The surface and the text editor have their own tests; here they only report what the window hands them.
vi.mock("@/components/ArchitectureDiagramSurface", () => ({
  ArchitectureDiagramSurface: ({ select, onRemoveFromView }: { select?: string | null, onRemoveFromView?: (ids: string[]) => void }) => (
    <div data-testid="architecture-surface" data-select={select ?? ""}>
      <button type="button" onClick={() => onRemoveFromView?.(["backend.api"])}>remove-node</button>
    </div>
  ),
}))
vi.mock("@/components/DocumentEditor", () => ({
  DocumentEditor: ({ path, onSaved }: { path: string, onSaved?: () => void }) => <button type="button" data-testid="text-editor" onClick={() => onSaved?.()}>{path}</button>,
}))

const feed = vi.hoisted(() => ({ listener: null as ((event: ChangeEvent) => void) | null }))
vi.mock("@/lib/syncStream", () => ({
  subscribeToChanges: (listener: (event: ChangeEvent) => void) => {
    feed.listener = listener
    return () => { feed.listener = null }
  },
}))
vi.mock("@/lib/deviceId", () => ({ getDeviceId: () => "this-device" }))
/** Another device edited one of the project's architecture files. */
const remoteChange = () => act(() => feed.listener!({ seq: 1, resource_kind: "document", resource_key: "graph/proj/backend/backend.c4", version: 2, device_id: "other-device" }))

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
}))
vi.mock("@/lib/api", () => api)

import { ArchitectureWindow } from "@/components/ArchitectureWindow"
import { UndoDeleteProvider } from "@/contexts/UndoDeleteContext"

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
    act(() => { button("System").click() })
    await act(async () => { fireEvent.keyDown(view.getByLabelText("System name"), { key: "Enter", target: { value: title } }) })
  }
  return { ...view, button, createSystem }
}

describe("ArchitectureWindow", () => {
  beforeEach(() => { vi.resetAllMocks() })
  afterEach(() => { vi.useRealTimers() })

  it("opens on the first system and moves through its tree", async () => {
    const { button, createSystem, getByLabelText, findByRole, getByTestId } = await open()
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
    act(() => { button("Module").click() })
    await act(async () => { fireEvent.keyDown(getByLabelText("Module name"), { key: "Enter", target: { value: "DB" } }) })
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
    act(() => { getByLabelText("Delete Backend and its modules").click() })
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

  it("never lets an earlier read replace a later write's answer", async () => {
    const { createSystem, queryByText } = await open()
    const read = deferred<ArchitectureWorkspace>()
    const write = deferred<ArchitectureWorkspace>()
    api.readArchitecture.mockReturnValueOnce(read.promise)
    api.applyArchitectureOperations.mockReturnValueOnce(write.promise)

    remoteChange()
    await createSystem("Frontend")
    // The write's answer arrives first, the earlier read's after it.
    await act(async () => {
      write.resolve(retitled("Written"))
      read.resolve(retitled("Read"))
    })
    await waitFor(() => expect(queryByText("Written")).toBeTruthy())
    expect(queryByText("Read")).toBeNull()
  })

  it("rereads once after a write when another device changed the model meanwhile", async () => {
    const { createSystem, findByText } = await open()
    const write = deferred<ArchitectureWorkspace>()
    api.applyArchitectureOperations.mockReturnValueOnce(write.promise)
    api.readArchitecture.mockResolvedValueOnce(retitled("Changed elsewhere"))

    await createSystem("Frontend")
    remoteChange()
    remoteChange()
    await act(async () => {})
    expect(api.readArchitecture).toHaveBeenCalledTimes(1)

    await act(async () => { write.resolve(workspace) })
    await findByText("Changed elsewhere")
    expect(api.readArchitecture).toHaveBeenCalledTimes(2)
  })

  it("shows a failed reread until a later one succeeds", async () => {
    const { button, getByTestId, findByRole, queryByRole } = await open()
    act(() => { button("Text").click() })

    // Saving text rereads the model.
    api.readArchitecture.mockRejectedValueOnce(new Error("Storage unavailable"))
    act(() => { getByTestId("text-editor").click() })
    expect((await findByRole("alert")).textContent).toBe("Storage unavailable")

    api.readArchitecture.mockResolvedValueOnce(workspace)
    remoteChange()
    await waitFor(() => expect(queryByRole("alert")).toBeNull())
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
