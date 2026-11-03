import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, render, screen } from "@testing-library/react"
import type { NeovimSelection } from "@/features/editor/NeovimEditor"
import type { ComposerInsertion } from "@/features/omp/OmpComposer"
import type { FileChat } from "@/features/omp/useFileChat"

type Sessions = { active: string | null; sessionIds: string[]; moved: string[] }
const api = vi.hoisted(() => ({
  loadFileSessions: vi.fn<(path: string) => Promise<Sessions>>(),
  saveFileSession: vi.fn(),
  informFileSession: vi.fn(),
  fetchNotebookDirectory: vi.fn<(path: string) => Promise<string>>(),
}))
const neovim = vi.hoisted(() => ({ onSelection: undefined as ((selection: NeovimSelection) => void) | undefined }))
vi.mock("@/lib/backend/api", () => api)
// The notebook and the omp frame are out of scope: the notebook says which file it shows, the panel what its chat would open and its draft
vi.mock("@/features/editor/NotebookNeovim", () => ({
  NotebookWithNeovim: ({ path, onSelection }: { path: string; onSelection?: (selection: NeovimSelection) => void }) => {
    neovim.onSelection = onSelection
    return <p>notebook {path}</p>
  },
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

beforeEach(() => {
  api.fetchNotebookDirectory.mockResolvedValue(MOUNT)
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

const workbenchOf = (drawerOpen: boolean, onDrawerOpenChange = vi.fn()) => (
  <NotebookWorkbench
    path={PATH}
    tabId="tab"
    ompView={undefined}
    onViewChange={vi.fn()}
    drawerOpen={drawerOpen}
    onDrawerOpenChange={onDrawerOpenChange}
    shellCommands={{} as never}
    modelProps={{} as never}
  />
)

describe("NotebookWorkbench", () => {
  it("links the chat to the notebook: a new session starts in the mount root holding it, so relative links resolve there, and names the file relative to it", async () => {
    api.loadFileSessions.mockResolvedValue({ active: null, sessionIds: [], moved: [] })
    render(workbenchOf(true))
    expect(await screen.findByText(`notebook ${PATH}`)).toBeTruthy()
    expect(await screen.findByText(`chat ${JSON.stringify({ kind: "new", cwd: MOUNT })} File: notebooks/analysis.ipynb`)).toBeTruthy()
    expect(api.fetchNotebookDirectory).toHaveBeenCalledWith(PATH, expect.any(AbortSignal))
    expect(api.loadFileSessions).toHaveBeenCalledWith(PATH)
  })

  it("starts a notebook outside every mount in its folder, the directory the backend answers", async () => {
    api.fetchNotebookDirectory.mockResolvedValue("/home/me/project/notebooks")
    api.loadFileSessions.mockResolvedValue({ active: null, sessionIds: [], moved: [] })
    render(workbenchOf(true))
    expect(await screen.findByText(`chat ${JSON.stringify({ kind: "new", cwd: "/home/me/project/notebooks" })} File: analysis.ipynb`)).toBeTruthy()
  })

  it("offers a retry when the directory cannot be found", async () => {
    api.fetchNotebookDirectory.mockRejectedValueOnce(new Error("offline"))
    api.loadFileSessions.mockResolvedValue({ active: null, sessionIds: [], moved: [] })
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

  it("quotes lines picked with <leader>a into the draft as the notebook's, fenced in the language of the cell's buffer, opening the drawer", async () => {
    api.loadFileSessions.mockResolvedValue({ active: null, sessionIds: [], moved: [] })
    const onDrawerOpenChange = vi.fn()
    const { rerender } = render(workbenchOf(false, onDrawerOpenChange))
    await screen.findByText(`notebook ${PATH}`)
    act(() => neovim.onSelection?.({ path: `${PATH}.py`, start_line: 2, end_line: 3, text: "x = 1\ny = 2" }))
    expect(onDrawerOpenChange).toHaveBeenCalledWith(true)

    rerender(workbenchOf(true, onDrawerOpenChange))
    expect((await screen.findByTestId("draft")).textContent).toBe("analysis.ipynb:2-3\n```python\nx = 1\ny = 2\n```\n\n")
    act(() => neovim.onSelection?.({ path: `${PATH}.md`, start_line: 1, end_line: 1, text: "# Title" }))
    expect(screen.getByTestId("draft").textContent).toBe("analysis.ipynb:1-1\n```markdown\n# Title\n```\n\n")
  })
})
