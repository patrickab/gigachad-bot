import { renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { onFileChange } from "@/lib/backend/fileEvents"
import { useNeovimTerminal, type NeovimDocument } from "../NeovimEditor"

vi.mock("@xterm/xterm/css/xterm.css", () => ({}))
vi.mock("@xterm/xterm", () => ({
  Terminal: class {
    rows = 24
    cols = 80
    options = {}
    loadAddon() {}
    onData() {}
    onBinary() {}
    onResize() {}
    open() {}
    write() {}
    resize() {}
    focus() {}
    dispose() {}
  },
}))
vi.mock("@xterm/addon-fit", () => ({ FitAddon: class { fit() {}; proposeDimensions() { return undefined } } }))
vi.mock("@xterm/addon-webgl", () => ({ WebglAddon: class { onContextLoss() {}; dispose() {} } }))
vi.mock("@/lib/backend/urls", () => ({ neovimSessionUrl: () => "ws://backend/neovim/session" }))
const api = vi.hoisted(() => ({
  storeWorkbenchImage: vi.fn<(path: string, image: Blob) => Promise<{ path: string; reference: string; changed: string[] }>>(),
}))
vi.mock("@/lib/backend/api", () => api)

class FakeSocket {
  static readonly CONNECTING = 0
  static readonly OPEN = 1
  static instances: FakeSocket[] = []
  readyState = FakeSocket.CONNECTING
  binaryType = ""
  sent: string[] = []
  onopen: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  onclose: (() => void) | null = null

  constructor(readonly url: string) {
    FakeSocket.instances.push(this)
  }

  send(data: unknown) { if (typeof data === "string") this.sent.push(data) }
  close() {}

  open() {
    this.readyState = FakeSocket.OPEN
    this.onopen?.()
  }

  receive(message: object) { this.onmessage?.({ data: JSON.stringify(message) }) }

  get frames(): Record<string, unknown>[] { return this.sent.map((frame) => JSON.parse(frame)) }
}

const firstDocument: NeovimDocument = { key: "session", path: "/p/main.tex", filename: "main.tex", content: "text" }

interface FakeItem {
  kind: string
  type: string
  getAsFile: () => File | null
}

async function connect(options: Parameters<typeof useNeovimTerminal>[3] = {}, save = vi.fn<(text: string, path: string) => Promise<string>>(async (text) => text)) {
  const view = renderHook(
    ({ document }: { document: NeovimDocument }) => useNeovimTerminal(document, save, vi.fn(), options),
    { initialProps: { document: firstDocument } },
  )
  await waitFor(() => expect(view.result.current.terminal).not.toBeNull())
  return { ...view, socket: FakeSocket.instances[0], save }
}

beforeEach(() => {
  api.storeWorkbenchImage.mockReset()
  vi.stubGlobal("WebSocket", FakeSocket)
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null)
})

afterEach(() => {
  FakeSocket.instances = []
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe("useNeovimTerminal frames", () => {
  it("keeps openFile and renameFile called while connecting for after the open frame, the others need an open socket", async () => {
    const { result, socket } = await connect()

    expect(result.current.openFile("/p/chapter.tex")).toBe(true)
    expect(result.current.renameFile("/p/main.tex", "/p/paper.tex")).toBe(true)
    expect(result.current.insertLines("x")).toBe(false)
    socket.open()

    expect(socket.frames.map((frame) => frame.type)).toEqual(["open", "open_file", "renamed"])
    expect(socket.frames[0]).toMatchObject({ path: "/p/main.tex", filename: "main.tex", content: "text" })
    expect(socket.frames[2]).toEqual({ type: "renamed", from: "/p/main.tex", path: "/p/paper.tex" })
  })

  it("sends the latest text of each path pushed while the session opens once it is open, then sends at once", async () => {
    const { result, socket } = await connect()

    expect(result.current.setContent("first", "/p/main.tex")).toBe(true)
    expect(result.current.setContent("second", "/p/main.tex")).toBe(true)
    expect(result.current.setContent("chapter", "/p/chapter.tex")).toBe(true)
    expect(socket.frames).toEqual([])
    socket.open()
    expect(result.current.setContent("third", "/p/main.tex")).toBe(true)

    expect(socket.frames.slice(1)).toEqual([
      { type: "set_content", content: "second", path: "/p/main.tex" },
      { type: "set_content", content: "chapter", path: "/p/chapter.tex" },
      { type: "set_content", content: "third", path: "/p/main.tex" },
    ])
  })

  it("takes no text without a session", () => {
    const { result } = renderHook(() => useNeovimTerminal(null, vi.fn(), vi.fn()))

    expect(result.current.setContent("text", "/p/main.tex")).toBe(false)
  })

  it("renames the buffer of the previous path when the path of the document changes under the same key", async () => {
    const { rerender, socket } = await connect()
    socket.open()

    rerender({ document: { ...firstDocument, path: "/p/other.tex", filename: "other.tex" } })

    expect(socket.frames.at(-1)).toEqual({ type: "renamed", from: "/p/main.tex", path: "/p/other.tex" })
  })

  it("saves with the path the save frame names", async () => {
    const { save, socket } = await connect()
    socket.open()

    socket.receive({ type: "save", id: 7, path: "/p/chapter.tex", content: "body" })

    await waitFor(() => expect(socket.frames.at(-1)).toEqual({ type: "saved", id: 7, content: "body" }))
    expect(save).toHaveBeenCalledWith("body", "/p/chapter.tex")
  })
})

describe("clipboard paste", () => {
  const screenshot = new File(["png"], "shot.png", { type: "image/png" })
  const reference = "\\includegraphics[width=\\linewidth]{imgs/img-20260101-120000.png}"
  const stored = { path: "/p/imgs/img-20260101-120000.png", reference, changed: ["/p/main.tex"] }
  const item = (kind: string, type: string, file: File | null = null) => ({ kind, type, getAsFile: () => file })
  const clipboardItem = (parts: Record<string, Blob>) => ({ types: Object.keys(parts), getType: async (type: string) => parts[type] })

  /** Open the session, answer `clipboard.read()` with *items* (or *denied*) and ask for a paste at *path*. */
  async function readRequest(path: string, items: ReturnType<typeof clipboardItem>[], denied = false) {
    const read = vi.fn(async () => { if (denied) throw new DOMException("denied", "NotAllowedError"); return items })
    vi.stubGlobal("navigator", { clipboard: { read } })
    const { socket } = await connect()
    socket.open()
    socket.receive({ type: "clipboard_read", id: 3, path })
    await waitFor(() => expect(socket.frames.at(-1)).toMatchObject({ type: "clipboard", id: 3 }))
    return socket
  }

  it("stores a PNG of the clipboard and answers its reference with a newline", async () => {
    api.storeWorkbenchImage.mockResolvedValueOnce(stored)
    const heard = vi.fn()
    const stop = onFileChange(() => true, heard)
    const socket = await readRequest("/p/main.tex", [clipboardItem({ "image/png": screenshot })])

    expect(api.storeWorkbenchImage).toHaveBeenCalledExactlyOnceWith("/p/main.tex", screenshot)
    expect(socket.frames.at(-1)).toEqual({ type: "clipboard", id: 3, text: `${reference}\n` })
    expect(heard.mock.calls).toEqual([["/p/imgs/img-20260101-120000.png"], ["/p/main.tex"]])
    stop()
  })

  it("answers the text of the clipboard, also when a picture comes with it", async () => {
    const text = Object.assign(new Blob(["a\tb"]), { text: async () => "a\tb" })
    const socket = await readRequest("/p/main.tex", [clipboardItem({ "text/plain": text, "image/png": screenshot })])

    expect(socket.frames.at(-1)).toEqual({ type: "clipboard", id: 3, text: "a\tb" })
    expect(api.storeWorkbenchImage).not.toHaveBeenCalled()
  })

  it("answers an error when the clipboard permission is denied", async () => {
    const socket = await readRequest("/p/main.tex", [], true)

    expect(socket.frames.at(-1)).toEqual({ type: "clipboard", id: 3, error: expect.any(String) })
  })

  it.each(["/p/notes.txt", "project/notes.md"])("answers an error for an image into %s without uploading", async (path) => {
    const socket = await readRequest(path, [clipboardItem({ "image/png": screenshot })])

    expect(socket.frames.at(-1)).toEqual({ type: "clipboard", id: 3, error: expect.any(String) })
    expect(api.storeWorkbenchImage).not.toHaveBeenCalled()
  })

  it("answers the reason when the image cannot be stored", async () => {
    api.storeWorkbenchImage.mockRejectedValueOnce(new Error("Image is too large"))
    const socket = await readRequest("/p/main.tex", [clipboardItem({ "image/png": screenshot })])

    expect(socket.frames.at(-1)).toEqual({ type: "clipboard", id: 3, error: "Image is too large" })
  })

  describe("paste event", () => {
    /** Mount the terminal and paste *items* into its host element. */
    async function paste(items: FakeItem[], options: Parameters<typeof useNeovimTerminal>[3] = {}) {
      const { result, socket } = await connect(options)
      socket.open()
      const parent = window.document.createElement("div")
      result.current.terminal?.attach(parent)
      const event = new Event("paste", { bubbles: true, cancelable: true })
      Object.assign(event, { clipboardData: { items } })
      parent.firstElementChild?.dispatchEvent(event)
      return { event, socket }
    }

    it("stores a pasted PNG and inserts its reference below the cursor", async () => {
      api.storeWorkbenchImage.mockResolvedValueOnce(stored)
      const { event, socket } = await paste([item("file", "image/png", screenshot)])

      expect(event.defaultPrevented).toBe(true)
      await waitFor(() => expect(socket.frames.at(-1)).toEqual({ type: "insert_lines", text: reference }))
      expect(api.storeWorkbenchImage).toHaveBeenCalledExactlyOnceWith("/p/main.tex", screenshot)
    })

    it("stores a pasted PNG next to the buffer on screen", async () => {
      api.storeWorkbenchImage.mockResolvedValueOnce(stored)
      await paste([item("file", "image/png", screenshot)], { shownPath: () => "/p/chapter.tex" })

      await waitFor(() => expect(api.storeWorkbenchImage).toHaveBeenCalledExactlyOnceWith("/p/chapter.tex", screenshot))
    })

    it("sends a message when the image is too large", async () => {
      api.storeWorkbenchImage.mockRejectedValueOnce(new Error("Image is too large"))
      const { socket } = await paste([item("file", "image/png", screenshot)])

      await waitFor(() => expect(socket.frames.at(-1)).toEqual({ type: "message", text: "Image is too large" }))
      expect(socket.frames.map((frame) => frame.type)).not.toContain("insert_lines")
    })

    it.each([
      ["a PNG that comes with text", [item("string", "text/plain"), item("file", "image/png", screenshot)]],
      ["a JPEG", [item("file", "image/jpeg", new File(["jpg"], "shot.jpg", { type: "image/jpeg" }))]],
    ])("leaves %s to the terminal", async (_name, items) => {
      const { event } = await paste(items)

      expect(event.defaultPrevented).toBe(false)
      expect(api.storeWorkbenchImage).not.toHaveBeenCalled()
    })
  })
})
