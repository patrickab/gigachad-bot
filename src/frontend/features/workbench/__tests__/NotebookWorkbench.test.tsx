import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, render, screen } from "@testing-library/react"
import { useEffect } from "react"
import type { NeovimDocument } from "@/features/editor/NeovimEditor"
import type { NeovimCommand, NeovimEventHandlers } from "@/features/editor/neovimProtocol"
import { FakeNotebook } from "@/features/notebook/__tests__/fakeNotebook"
import type { NotebookTextStore } from "@/features/notebook/notebookTextStore"
import { NOTEBOOK_TEXT_DELAY_MS } from "@/features/notebook/notebookTextStore"
import type { ComposerInsertion } from "@/features/omp/OmpComposer"
import type { FileChat } from "@/features/omp/useFileChat"

type Sessions = { active: string | null; sessionIds: string[]; moved: string[] }
const api = vi.hoisted(() => ({
  loadFileSessions: vi.fn<(path: string) => Promise<Sessions>>(),
  saveFileSession: vi.fn(),
  informFileSession: vi.fn(),
  fetchNotebookDirectory: vi.fn<(path: string) => Promise<string>>(),
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
vi.mock("@/features/omp/OmpChatPanel", () => ({
  OmpChatPanel: ({ chat, insertion }: { chat: FileChat; insertion?: ComposerInsertion | null }) => <>
    <p>chat {JSON.stringify(chat.view)} {chat.lead}</p>
    {insertion && <pre data-testid="draft">{insertion.text}</pre>}
  </>,
}))

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
  api.fetchNotebookDirectory.mockResolvedValue(MOUNT)
  api.loadFileSessions.mockResolvedValue({ active: null, sessionIds: [], moved: [] })
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
    ompView={undefined}
    onViewChange={vi.fn()}
    drawerOpen={drawerOpen}
    onDrawerOpenChange={onDrawerOpenChange}
    shellCommands={{} as never}
    modelProps={{} as never}
  />
)

describe("NotebookWorkbench chat", () => {
  it("links the chat to the notebook: a new session starts in the mount root holding it, so relative links resolve there, and names the file relative to it", async () => {
    render(workbenchOf(true))
    expect(await screen.findByText(`notebook ${PATH}`)).toBeTruthy()
    expect(await screen.findByText(`chat ${JSON.stringify({ kind: "new", cwd: MOUNT })} File: notebooks/analysis.ipynb`)).toBeTruthy()
    expect(api.fetchNotebookDirectory).toHaveBeenCalledWith(PATH, expect.any(AbortSignal))
    expect(api.loadFileSessions).toHaveBeenCalledWith(PATH)
  })

  it("starts a notebook outside every mount in its folder, the directory the backend answers", async () => {
    api.fetchNotebookDirectory.mockResolvedValue("/home/me/project/notebooks")
    render(workbenchOf(true))
    expect(await screen.findByText(`chat ${JSON.stringify({ kind: "new", cwd: "/home/me/project/notebooks" })} File: analysis.ipynb`)).toBeTruthy()
  })

  it("offers a retry when the directory cannot be found", async () => {
    api.fetchNotebookDirectory.mockRejectedValueOnce(new Error("offline"))
    render(workbenchOf(true))
    const retry = await screen.findByRole("button", { name: /retry/i })
    act(() => retry.click())
    expect(await screen.findByText(`notebook ${PATH}`)).toBeTruthy()
  })

  it("shows no chat while the drawer is closed", async () => {
    api.loadFileSessions.mockResolvedValue({ active: "linked", sessionIds: ["linked"], moved: [] })
    render(workbenchOf(false))
    expect(await screen.findByText(`notebook ${PATH}`)).toBeTruthy()
    await vi.waitFor(() => expect(api.loadFileSessions).toHaveBeenCalled())
    expect(screen.queryByText(/^chat /)).toBeNull()
  })

  it("quotes lines picked with <leader>a into the draft as the notebook's, fenced in the language of the buffer, opening the drawer", async () => {
    notebook.model = notebookOf("x = 1\ny = 2")
    const onDrawerOpenChange = vi.fn()
    const { rerender } = render(workbenchOf(false, onDrawerOpenChange))
    await screen.findByText(`notebook ${PATH}`)
    act(() => neovim.options.on?.selection?.({ type: "selection", path: `${PATH}.py`, start_line: 2, end_line: 3, text: "x = 1\ny = 2" }))
    expect(onDrawerOpenChange).toHaveBeenCalledWith(true)

    rerender(workbenchOf(true, onDrawerOpenChange))
    expect((await screen.findByTestId("draft")).textContent).toBe("analysis.ipynb:2-3\n```python\nx = 1\ny = 2\n```\n\n")
  })
})

describe("NotebookWorkbench Neovim", () => {
  it("shows the notebook alone while Neovim is off", async () => {
    neovim.enabled = false
    notebook.model = notebookOf("x = 1")
    render(workbenchOf(false))
    await screen.findByText(`notebook ${PATH}`)
    expect(screen.queryByText("neovim")).toBeNull()
    expect(neovim.documents.every((document) => document === null)).toBe(true)
  })

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

  it("hands every change of the notebook to the buffer", async () => {
    const model = notebookOf("x = 1")
    notebook.model = model
    render(workbenchOf(false))
    await screen.findByText("neovim")
    vi.useFakeTimers()
    act(() => {
      model.edit(0, "x = 3")
      vi.advanceTimersByTime(NOTEBOOK_TEXT_DELAY_MS)
    })
    expect(neovim.send).toHaveBeenCalledWith({ type: "set_content", content: "# %%\nx = 3\n", path: `${PATH}.py` })
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

  it("opens Neovim again for another notebook opened in the tab after :q", async () => {
    notebook.model = notebookOf("x = 1")
    const { rerender } = render(workbenchOf(false))
    await screen.findByText("neovim")
    act(() => neovim.quit?.())
    expect(screen.getByRole("button", { name: "Start Neovim" })).toBeTruthy()

    const other = "/home/me/project/notebooks/other.ipynb"
    rerender(workbenchOf(false, undefined, other))
    expect(neovim.documents.at(-1)).toMatchObject({ key: "tab:1", path: `${other}.py`, notebook: true })
    expect(screen.queryByRole("button", { name: "Start Neovim" })).toBeNull()
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
