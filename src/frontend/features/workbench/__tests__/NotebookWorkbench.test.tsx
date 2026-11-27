import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, render, screen } from "@testing-library/react"
import { useEffect } from "react"
import type { NeovimDocument } from "@/features/editor/NeovimEditor"
import type { NeovimCommand, NeovimEventHandlers } from "@/features/editor/neovimProtocol"
import { FakeNotebook } from "@/features/notebook/__tests__/fakeNotebook"
import type { NotebookTextStore } from "@/features/notebook/notebookTextStore"
import { NOTEBOOK_TEXT_DELAY_MS } from "@/features/notebook/notebookTextStore"
import type { OpenedWorkbenchChat } from "@/lib/backend/api"
import { openedChat } from "./chatFixtures"

const api = vi.hoisted(() => ({
  openWorkbenchChat: vi.fn<(request: { path: string; sessionId?: string }) => Promise<OpenedWorkbenchChat>>(),
  markWorkbenchChatInformed: vi.fn(),
  ApiError: class extends Error { status = 0 },
}))
interface NeovimOptions { askKeymap?: boolean; on?: NeovimEventHandlers }
const neovim = vi.hoisted(() => ({
  enabled: true,
  documents: [] as (NeovimDocument | null)[],
  save: null as ((text: string, path: string) => Promise<string>) | null,
  quit: null as (() => void) | null,
  options: {} as NeovimOptions,
  send: vi.fn<(command: NeovimCommand) => boolean>(() => true),
}))
const notebook = vi.hoisted(() => ({ model: null as FakeNotebook | null, persist: vi.fn(async () => true), run: vi.fn() }))
vi.mock("@/lib/backend/api", () => api)
vi.mock("@/contexts/SettingsContext", () => ({ useNeovimSetting: () => ({ enabled: neovim.enabled }) }))
// Neovim notes the sessions it is asked for, the notebook binds `notebook.model` into the store once mounted
vi.mock("@/features/editor/NeovimEditor", () => ({
  NeovimEditor: () => <p>neovim</p>,
  NeovimStopped: ({ action, onAction }: { action: string; onAction: () => void }) => <button onClick={onAction}>{action}</button>,
  useNeovimTerminal: (document: NeovimDocument | null, save: (text: string, path: string) => Promise<string>, quit: () => void, options: NeovimOptions) => {
    neovim.documents.push(document)
    neovim.save = save
    neovim.quit = quit
    neovim.options = options
    return { terminal: null, status: { state: "running" }, restart: () => {}, send: neovim.send }
  },
}))
// Keyed by the generation like the real one, so another notebook in the tab mounts afresh and binds `notebook.model` as it is then
function FakeNotebookEditor({ path, textStore }: { path: string; textStore: NotebookTextStore }) {
  useEffect(() => (notebook.model ? textStore.bind(notebook.model, notebook.persist, notebook.run) : undefined), [textStore])
  return <p>notebook {path}</p>
}
vi.mock("@/features/notebook/NotebookDocument", () => ({
  // Another notebook opened in the tab is the next generation
  useEditorPath: (path: string) => ({ path, generation: path === PATH ? 0 : 1 }),
  NotebookDocument: ({ editor, textStore }: { editor: { path: string; generation: number }; textStore: NotebookTextStore }) => (
    <FakeNotebookEditor key={editor.generation} path={editor.path} textStore={textStore} />
  ),
}))
// vi.mock is hoisted above the static imports, so the factory loads the stand-in itself
vi.mock("../DocumentChat", async () => ({ DocumentChat: (await import("./chatFixtures")).FakeDocumentChat }))

import { NotebookWorkbench } from "../NotebookWorkbench"

const PATH = "/home/me/project/notebooks/analysis.ipynb"
const MOUNT = "/home/me/project"

/** A notebook of one code cell `source`, its buffer extension declared as `extension`. */
const notebookOf = (source: string, extension?: string) => new FakeNotebook({
  cells: [{ cell_type: "code", id: "a", source, metadata: {}, outputs: [], execution_count: null }],
  metadata: extension ? { language_info: { file_extension: extension } } : {},
  nbformat: 4,
  nbformat_minor: 5,
})

beforeEach(() => {
  api.openWorkbenchChat.mockResolvedValue(openedChat({ id: "s1", cwd: MOUNT, kind: "notebook" }))
  neovim.enabled = true
  notebook.model = null
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  vi.useRealTimers()
  neovim.documents = []
})

const workbenchOf = (drawerOpen: boolean, onDrawerOpenChange = vi.fn(), path = PATH) => (
  <NotebookWorkbench
    path={path}
    tabId="tab"
    chatSessionId={undefined}
    onChatSessionChange={vi.fn()}
    drawerOpen={drawerOpen}
    onDrawerOpenChange={onDrawerOpenChange}
    modelProps={{} as never}
  />
)

describe("NotebookWorkbench Neovim", () => {
  it("opens the notebook's text beside it once the model is bound, saves and runs cells through the store", async () => {
    const model = notebookOf("x = 1")
    notebook.model = model
    render(workbenchOf(false))
    await screen.findByText("neovim")
    expect(neovim.documents.filter((document) => document !== null).at(-1)).toEqual({ key: "tab:0", path: `${PATH}.py`, filename: "analysis.ipynb.py", content: "# %%\nx = 1\n", notebook: true })
    expect(neovim.options.askKeymap).toBe(true)

    await expect(neovim.save?.("# %%\nx = 2\n\n# %%\ny = 1\n", `${PATH}.py`)).resolves.toBe("# %%\nx = 2\n\n# %%\ny = 1\n")
    expect(model.cells.map((cell) => cell.source)).toEqual(["x = 2", "y = 1"])
    expect(notebook.persist).toHaveBeenCalledTimes(1)
    neovim.options.on?.run_cell?.({ type: "run_cell", path: `${PATH}.py`, line: 4 })
    expect(notebook.run).toHaveBeenCalledWith(1)
  })

  it("leaves the notebook alone after :q and starts a fresh session on its current text, edits made while closed included", async () => {
    const model = notebookOf("x = 1")
    notebook.model = model
    render(workbenchOf(false))
    await screen.findByText("neovim")
    act(() => neovim.quit?.())
    expect(neovim.documents.at(-1)).toBeNull()
    expect(screen.getByText(`notebook ${PATH}`)).toBeTruthy()

    vi.useFakeTimers()
    act(() => {
      model.edit(0, "x = 4")
      vi.advanceTimersByTime(NOTEBOOK_TEXT_DELAY_MS)
    })
    const opened = neovim.documents.length
    act(() => screen.getByRole("button", { name: "Start Neovim" }).click())
    const reopened = neovim.documents.slice(opened)
    expect(reopened.length).toBeGreaterThan(0)
    expect(reopened.every((document) => document?.path === `${PATH}.py` && document.content === "# %%\nx = 4\n" && document.notebook)).toBe(true)
    expect(screen.getByText("neovim")).toBeTruthy()
  })

  it("never opens a session for another notebook with the previous notebook's text", async () => {
    notebook.model = notebookOf("old = 1")
    const { rerender } = render(workbenchOf(false))
    await screen.findByText("neovim")

    const other = "/home/me/project/notebooks/other.ipynb"
    notebook.model = notebookOf("new = 1", ".jl")
    rerender(workbenchOf(false, undefined, other))
    const opened = neovim.documents.filter((document) => document?.key === "tab:1")
    expect(opened.length).toBeGreaterThan(0)
    expect(opened.every((document) => document?.path === `${other}.jl` && document.content === "# %%\nnew = 1\n")).toBe(true)
  })
})
