import { renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
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
    expect(result.current.setContent("new", "/p/main.tex")).toBe(false)
    expect(result.current.insertLines("x")).toBe(false)
    socket.open()

    expect(socket.frames.map((frame) => frame.type)).toEqual(["open", "open_file", "renamed"])
    expect(socket.frames[0]).toMatchObject({ path: "/p/main.tex", filename: "main.tex", content: "text" })
    expect(socket.frames[2]).toEqual({ type: "renamed", from: "/p/main.tex", path: "/p/paper.tex" })
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

describe("image paste", () => {
  const screenshot = new File(["png"], "shot.png", { type: "image/png" })
  const item = (kind: string, type: string, file: File | null = null) => ({ kind, type, getAsFile: () => file })

  /** Mount the terminal and paste *items* into its host element. */
  async function paste(items: FakeItem[], onPasteImage?: (image: Blob) => Promise<string | null>) {
    const { result, socket } = await connect({ onPasteImage })
    socket.open()
    const parent = window.document.createElement("div")
    result.current.terminal?.attach(parent)
    const event = new Event("paste", { bubbles: true, cancelable: true })
    Object.assign(event, { clipboardData: { items } })
    parent.firstElementChild?.dispatchEvent(event)
    return { event, socket }
  }

  it("takes a pasted PNG and inserts the returned text below the cursor", async () => {
    const onPasteImage = vi.fn(async () => "\\includegraphics[width=\\linewidth]{imgs/shot.png}")
    const { event, socket } = await paste([item("file", "image/png", screenshot)], onPasteImage)

    expect(event.defaultPrevented).toBe(true)
    expect(onPasteImage).toHaveBeenCalledWith(screenshot)
    await waitFor(() => expect(socket.frames.at(-1)).toEqual({ type: "insert_lines", text: "\\includegraphics[width=\\linewidth]{imgs/shot.png}" }))
  })

  it("inserts nothing when the callback gives no text", async () => {
    const onPasteImage = vi.fn(async () => null)
    const { event, socket } = await paste([item("file", "image/png", screenshot)], onPasteImage)

    expect(event.defaultPrevented).toBe(true)
    await waitFor(() => expect(onPasteImage).toHaveBeenCalled())
    expect(socket.frames.map((frame) => frame.type)).not.toContain("insert_lines")
  })

  it.each([
    ["a PNG that comes with text", [item("string", "text/plain"), item("file", "image/png", screenshot)], true],
    ["a JPEG", [item("file", "image/jpeg", new File(["jpg"], "shot.jpg", { type: "image/jpeg" }))], true],
    ["a PNG without a callback", [item("file", "image/png", screenshot)], false],
  ])("leaves %s to the terminal", async (_name, items, withCallback) => {
    const onPasteImage = vi.fn(async () => "never")
    const { event } = await paste(items, withCallback ? onPasteImage : undefined)

    expect(event.defaultPrevented).toBe(false)
    expect(onPasteImage).not.toHaveBeenCalled()
  })
})
