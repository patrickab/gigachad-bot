import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, render, screen } from "@testing-library/react"
import { useEffect } from "react"
import type { NeovimDocument, NeovimSelection } from "@/features/editor/NeovimEditor"
import type { NotebookTextBridge } from "@/features/notebook/notebookTextBridge"
import type { ComposerInsertion } from "@/features/omp/OmpComposer"
import type { FileChat } from "@/features/omp/useFileChat"

type Sessions = { active: string | null; sessionIds: string[]; moved: string[] }
const api = vi.hoisted(() => ({
  loadFileSessions: vi.fn<(path: string) => Promise<Sessions>>(),
  saveFileSession: vi.fn(),
  informFileSession: vi.fn(),
  fetchNotebookDirectory: vi.fn<(path: string) => Promise<string>>(),
}))
interface NeovimOptions { askKeymap?: boolean; onSelection?: (selection: NeovimSelection) => void; onRunCell?: (path: string, line: number) => void }
const neovim = vi.hoisted(() => ({
  enabled: true,
  documents: [] as (NeovimDocument | null)[],
  save: null as ((text: string, path: string) => Promise<string>) | null,
  quit: null as (() => void) | null,
  options: {} as NeovimOptions,
  setContent: vi.fn(() => true),
}))
const notebook = vi.hoisted(() => ({ bridge: null as NotebookTextBridge | null }))
vi.mock("@/lib/backend/api", () => api)
vi.mock("@/contexts/SettingsContext", () => ({ useNeovimSetting: () => ({ enabled: neovim.enabled }) }))
// Neovim notes the sessions it is asked for, the notebook hands over `notebook.bridge` once mounted
vi.mock("@/features/editor/NeovimEditor", () => ({
  NeovimEditor: () => <p>neovim</p>,
  useNeovimTerminal: (document: NeovimDocument | null, save: (text: string, path: string) => Promise<string>, quit: () => void, options: NeovimOptions) => {
    neovim.documents.push(document)
    neovim.save = save
    neovim.quit = quit
    neovim.options = options
    return { terminal: null, status: { state: "running" }, restart: () => {}, setContent: neovim.setContent }
  },
}))
// Keyed by the generation like the real one, so another notebook in the tab mounts afresh and announces `notebook.bridge` as it is then
function FakeNotebookEditor({ path, onTextBridge }: { path: string; onTextBridge: (bridge: NotebookTextBridge | null) => void }) {
  useEffect(() => {
    onTextBridge(notebook.bridge)
    return () => onTextBridge(null)
  }, [onTextBridge])
  return <p>notebook {path}</p>
}
vi.mock("@/features/notebook/NotebookDocument", () => ({
  // Another notebook opened in the tab is the next generation
  useEditorPath: (path: string) => ({ path, generation: path === PATH ? 0 : 1 }),
  NotebookDocument: ({ editor, onTextBridge }: { editor: { path: string; generation: number }; onTextBridge: (bridge: NotebookTextBridge | null) => void }) => (
    <FakeNotebookEditor key={editor.generation} path={editor.path} onTextBridge={onTextBridge} />
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

/** A bridge whose projection is `text`, `emit` plays a change of the notebook. */
function fakeBridge(text: string) {
  let listener: ((text: string) => void) | null = null
  const bridge = {
    extension: ".py",
    text: vi.fn(() => text),
    save: vi.fn(async (edited: string) => edited),
    runAt: vi.fn(),
    subscribe: vi.fn((next: (text: string) => void) => { listener = next; return () => { listener = null } }),
  }
  return { bridge, emit: (next: string) => { text = next; listener?.(next) } }
}

beforeEach(() => {
  api.fetchNotebookDirectory.mockResolvedValue(MOUNT)
  api.loadFileSessions.mockResolvedValue({ active: null, sessionIds: [], moved: [] })
  neovim.enabled = true
  notebook.bridge = null
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
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
    notebook.bridge = fakeBridge("# %%\nx = 1\ny = 2\n").bridge
    const onDrawerOpenChange = vi.fn()
    const { rerender } = render(workbenchOf(false, onDrawerOpenChange))
    await screen.findByText(`notebook ${PATH}`)
    act(() => neovim.options.onSelection?.({ path: `${PATH}.py`, start_line: 2, end_line: 3, text: "x = 1\ny = 2" }))
    expect(onDrawerOpenChange).toHaveBeenCalledWith(true)

    rerender(workbenchOf(true, onDrawerOpenChange))
    expect((await screen.findByTestId("draft")).textContent).toBe("analysis.ipynb:2-3\n```python\nx = 1\ny = 2\n```\n\n")
  })
})

describe("NotebookWorkbench Neovim", () => {
  it("shows the notebook alone while Neovim is off", async () => {
    neovim.enabled = false
    notebook.bridge = fakeBridge("# %%\nx = 1\n").bridge
    render(workbenchOf(false))
    await screen.findByText(`notebook ${PATH}`)
    expect(screen.queryByText("neovim")).toBeNull()
    expect(neovim.documents.every((document) => document === null)).toBe(true)
  })

  it("opens the notebook's text beside it once the bridge arrives, saves and runs cells through the bridge", async () => {
    const { bridge } = fakeBridge("# %%\nx = 1\n")
    notebook.bridge = bridge
    render(workbenchOf(false))
    await screen.findByText("neovim")
    expect(neovim.documents.filter((document) => document !== null).at(-1)).toEqual({ key: "tab:0", path: `${PATH}.py`, filename: "analysis.ipynb.py", content: "# %%\nx = 1\n", notebook: true })
    expect(neovim.options.askKeymap).toBe(true)

    await expect(neovim.save?.("# %%\nx = 2\n", `${PATH}.py`)).resolves.toBe("# %%\nx = 2\n")
    expect(bridge.save).toHaveBeenCalledWith("# %%\nx = 2\n")
    neovim.options.onRunCell?.(`${PATH}.py`, 2)
    expect(bridge.runAt).toHaveBeenCalledWith(2)
  })

  it("hands every change of the notebook to the buffer", async () => {
    const { bridge, emit } = fakeBridge("# %%\nx = 1\n")
    notebook.bridge = bridge
    render(workbenchOf(false))
    await screen.findByText("neovim")
    act(() => emit("# %%\nx = 3\n"))
    expect(neovim.setContent).toHaveBeenCalledWith("# %%\nx = 3\n", `${PATH}.py`)
  })

  it("leaves the notebook alone after :q and starts a fresh session from its text then", async () => {
    const { bridge, emit } = fakeBridge("# %%\nx = 1\n")
    notebook.bridge = bridge
    render(workbenchOf(false))
    await screen.findByText("neovim")
    act(() => neovim.quit?.())
    expect(neovim.documents.at(-1)).toBeNull()
    expect(screen.getByText(`notebook ${PATH}`)).toBeTruthy()

    emit("# %%\nx = 4\n")
    act(() => screen.getByRole("button", { name: "Start Neovim" }).click())
    expect(neovim.documents.at(-1)).toMatchObject({ path: `${PATH}.py`, content: "# %%\nx = 4\n", notebook: true })
    expect(screen.getByText("neovim")).toBeTruthy()
  })

  it("opens Neovim again for another notebook opened in the tab after :q", async () => {
    notebook.bridge = fakeBridge("# %%\nx = 1\n").bridge
    const { rerender } = render(workbenchOf(false))
    await screen.findByText("neovim")
    act(() => neovim.quit?.())
    expect(screen.getByRole("button", { name: "Start Neovim" })).toBeTruthy()

    const other = "/home/me/project/notebooks/other.ipynb"
    rerender(workbenchOf(false, undefined, other))
    expect(neovim.documents.at(-1)).toMatchObject({ key: "tab:1", path: `${other}.py`, notebook: true })
    expect(screen.queryByRole("button", { name: "Start Neovim" })).toBeNull()
  })

  it("never opens a session for another notebook with the previous notebook's bridge", async () => {
    notebook.bridge = fakeBridge("# %%\nold = 1\n").bridge
    const { rerender } = render(workbenchOf(false))
    await screen.findByText("neovim")

    const other = "/home/me/project/notebooks/other.ipynb"
    const next = fakeBridge("# %%\nnew = 1\n").bridge
    next.extension = ".jl"
    notebook.bridge = next
    rerender(workbenchOf(false, undefined, other))
    const opened = neovim.documents.filter((document) => document?.key === "tab:1")
    expect(opened.length).toBeGreaterThan(0)
    expect(opened.every((document) => document?.path === `${other}.jl` && document.content === "# %%\nnew = 1\n")).toBe(true)
  })
})
