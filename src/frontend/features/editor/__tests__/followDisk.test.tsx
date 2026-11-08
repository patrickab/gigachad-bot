import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, expect, it, vi } from "vitest"
import { FilePanelProvider, FilePanelSlot, useOpenFile } from "../FilePanel"
import { TextDocumentEditor } from "../TextDocumentEditor"

const api = vi.hoisted(() => ({
  loadFileViewerText: vi.fn<(path: string) => Promise<string>>(),
  readFileRootRendered: vi.fn(),
  writeRootFile: vi.fn(),
  writeDocument: vi.fn(),
  writeArchitectureSource: vi.fn(),
  ApiError: class ApiError extends Error {
    status = 500
  },
}))
vi.mock("@/lib/backend/api", () => api)
vi.mock("@/lib/backend/syncStream", () => ({ syncCursorReady: () => Promise.resolve(), subscribeToChanges: () => () => {} }))

// The follow primitive is real, from the watch socket's `changed` frame to the editor: only the WebSocket is faked.
vi.mock("@/lib/backend/urls", () => ({ fileWatchUrl: () => "ws://backend/files/watch" }))

class FakeSocket {
  static readonly CONNECTING = 0
  static readonly OPEN = 1
  static readonly CLOSED = 3
  static instances: FakeSocket[] = []
  readyState = FakeSocket.CONNECTING
  sent: string[] = []
  onopen: (() => void) | null = null
  onmessage: ((event: { data: unknown }) => void) | null = null
  onclose: (() => void) | null = null

  constructor(readonly url: string) { FakeSocket.instances.push(this) }

  send(data: string) { this.sent.push(data) }

  close() {
    this.readyState = FakeSocket.CLOSED
    this.onclose?.()
  }

  open() {
    this.readyState = FakeSocket.OPEN
    this.onopen?.()
  }

  get watched(): { files: string[]; folders: string[] } | undefined {
    const last = this.sent.at(-1)
    return last ? JSON.parse(last) : undefined
  }
}

const neovim = vi.hoisted(() => ({ enabled: false, setContent: vi.fn() }))
vi.mock("@/contexts/SettingsContext", () => ({ useNeovimSetting: () => ({ available: true, enabled: neovim.enabled, setPreferred: () => {} }) }))
vi.mock("../NeovimEditor", () => ({
  NeovimEditor: () => <div>neovim</div>,
  useNeovimTerminal: () => ({ terminal: null, status: { state: "running" }, restart: () => {}, send: (command: { type: string; content?: string; path?: string }) => command.type === "set_content" && neovim.setContent(command.content, command.path) }),
}))
vi.mock("../AssistedConsoleEditor", () => ({
  AssistedConsoleEditor: ({ value, onChange }: { value: string; onChange: (value: string) => void }) => (
    <textarea aria-label="Document text" value={value} onChange={(event) => onChange(event.target.value)} />
  ),
}))
vi.mock("@/components/content/LaTeXMarkdown", () => ({
  LaTeXMarkdown: ({ content }: { content: string }) => <output data-testid="rendered-markdown">{content}</output>,
}))
vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} })

const PATH = "/repo/notes.md"

/** The editor registered its file with the watch socket. Another writer then changes `path` on disk to `text` and the server reports it. */
async function changeOnDisk(path: string, text: string) {
  const socket = await waitFor(() => {
    const open = FakeSocket.instances.at(-1)
    expect(open).toBeDefined()
    expect(open!.readyState).not.toBe(FakeSocket.CLOSED)
    return open!
  })
  if (socket.readyState === FakeSocket.CONNECTING) socket.open()
  await waitFor(() => expect(socket.watched?.files).toContain(path))
  api.loadFileViewerText.mockResolvedValue(text)
  await act(async () => { socket.onmessage?.({ data: JSON.stringify({ type: "changed", path }) }) })
}

function OpenButton() {
  const openFile = useOpenFile()
  return <button onClick={() => openFile(PATH)}>open file</button>
}

async function openFilePanel() {
  render(<FilePanelProvider><OpenButton /><FilePanelSlot /></FilePanelProvider>)
  fireEvent.click(screen.getByRole("button", { name: "open file" }))
}

beforeEach(() => {
  vi.stubGlobal("WebSocket", FakeSocket)
  neovim.enabled = false
  neovim.setContent.mockReset()
  api.loadFileViewerText.mockReset().mockResolvedValue("one")
})
afterEach(async () => {
  cleanup()
  // Leaving the last editor closes the shared watch socket after its debounce, the next test starts without it
  await waitFor(() => expect(FakeSocket.instances.every((socket) => socket.readyState === FakeSocket.CLOSED)).toBe(true))
  FakeSocket.instances = []
})

it("a followed change reaches a clean plain editor", async () => {
  render(<TextDocumentEditor path={PATH} slug="" onClose={vi.fn()} />)
  expect(await screen.findByRole("textbox", { name: "Document text" })).toHaveValue("one")

  await changeOnDisk(PATH, "two")
  expect(screen.getByRole("textbox", { name: "Document text" })).toHaveValue("two")
})

it("a dirty plain editor keeps its text, and follows again once saved", async () => {
  api.writeRootFile.mockResolvedValue(undefined)
  render(<TextDocumentEditor path={PATH} slug="" onClose={vi.fn()} />)
  const textbox = await screen.findByRole("textbox", { name: "Document text" })
  fireEvent.change(textbox, { target: { value: "mine" } })

  await changeOnDisk(PATH, "theirs")
  expect(textbox).toHaveValue("mine")

  fireEvent.keyDown(document, { key: "s", ctrlKey: true })
  await waitFor(() => expect(api.writeRootFile).toHaveBeenCalledWith(PATH, "mine"))
  await waitFor(() => expect(screen.queryByText(/modified/)).toBeNull())
  await changeOnDisk(PATH, "theirs again")
  expect(textbox).toHaveValue("theirs again")
})

it("a Neovim document gets the followed text through setContent", async () => {
  neovim.enabled = true
  render(<TextDocumentEditor path={PATH} slug="" onClose={vi.fn()} />)
  await waitFor(() => expect(screen.getByText("neovim")).toBeInTheDocument())

  await changeOnDisk(PATH, "two")
  expect(neovim.setContent).toHaveBeenCalledWith("two", PATH)
})

it("a Neovim document is not pushed the text it already has", async () => {
  neovim.enabled = true
  render(<TextDocumentEditor path={PATH} slug="" onClose={vi.fn()} />)
  await waitFor(() => expect(screen.getByText("neovim")).toBeInTheDocument())

  await changeOnDisk(PATH, "one")
  expect(neovim.setContent).not.toHaveBeenCalled()
})

it("a followed change reaches the file panel's text view", async () => {
  await openFilePanel()
  expect(await screen.findByTestId("rendered-markdown")).toHaveTextContent("one")

  await changeOnDisk(PATH, "two")
  expect(screen.getByTestId("rendered-markdown")).toHaveTextContent("two")
})

it("the file panel's Neovim gets the followed text through setContent", async () => {
  neovim.enabled = true
  await openFilePanel()
  await waitFor(() => expect(screen.getByText("neovim")).toBeInTheDocument())

  await changeOnDisk(PATH, "two")
  expect(neovim.setContent).toHaveBeenCalledWith("two", PATH)
})
