import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react"
import { File as FileIcon } from "lucide-react"
import type { Attachment } from "@/lib/domain/types"
import { uploadFile } from "@/lib/backend/api"
import { ChatInput, type ChatContext } from "../ChatInput"

vi.mock("@/features/chat/EnabledToolsContext", () => ({ useEnabledTools: () => ({ enabledTools: [], toggleTool: () => {} }) }))
vi.mock("@/contexts/SettingsContext", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/contexts/SettingsContext")>(),
  useSettings: () => ({ ocrModel: "" }),
  useNeovimSetting: () => ({ available: false, enabled: false, setPreferred: () => {} }),
}))
vi.mock("@/lib/backend/api", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/lib/backend/api")>(),
  uploadFile: vi.fn(async () => ({ name: "drawing.jpg", mime: "image/jpeg", url: "/drawing.jpg" })),
}))
vi.mock("@/lib/canvasExport", () => ({ renderStrokesToJpeg: async () => new Blob(["jpg"], { type: "image/jpeg" }) }))
vi.mock("@/contexts/ProjectContext", () => ({
  useDisplayName: () => (slug: string | null, path: string) => slug === "proj" && path === "project/proj/document/plan.md" ? "Roadmap.md" : undefined,
}))

const notes = (content: string): Attachment => ({ name: "notes.md", mime: "text/markdown", url: "", content, active: true })

function renderComposer(context: ChatContext) {
  return render(<ChatInput chatId="c1" onSend={() => {}} context={context} />)
}

function contextOf(attachments: ChatContext["attachments"]): ChatContext {
  return { attachments, onToggleActive: vi.fn(), onRemove: vi.fn(), onOpen: vi.fn() }
}

describe("ChatInput context chips", () => {
  afterEach(cleanup)

  it("routes open, toggle and remove to the message that carries the attachment", () => {
    // The same file name attached to two messages: each chip acts on its own message.
    const second = notes("v2")
    const context = contextOf([{ messageIndex: 0, attachment: notes("v1") }, { messageIndex: 2, attachment: second }])
    renderComposer(context)

    fireEvent.click(screen.getAllByRole("button", { name: "Preview notes.md" })[1])
    fireEvent.click(screen.getAllByRole("button", { name: "Exclude notes.md from context" })[1])
    fireEvent.click(screen.getAllByRole("button", { name: "Remove notes.md" })[1])

    expect(context.onOpen).toHaveBeenCalledWith(2, second)
    expect(context.onToggleActive).toHaveBeenCalledWith(2, "notes.md")
    expect(context.onRemove).toHaveBeenCalledWith(2, "notes.md")
  })
})

describe("ChatInput Escape", () => {
  afterEach(() => { cleanup(); vi.useRealTimers() })

  it("closes an open completion menu first, then leaves the draft", async () => {
    vi.useFakeTimers()
    const complete = async () => [{ label: "a.ts", icon: FileIcon, from: 0, to: 2, insert: "@a.ts ", caretOffset: 6 }]
    render(<ChatInput chatId="c1" onSend={() => {}} complete={complete} />)
    const draft = screen.getByRole("textbox") as HTMLTextAreaElement
    draft.focus()
    fireEvent.change(draft, { target: { value: "@a", selectionStart: 2 } })
    await act(() => vi.advanceTimersByTimeAsync(100))
    expect(screen.getByRole("listbox")).toBeInTheDocument()

    fireEvent.keyDown(draft, { key: "Escape" })
    expect(screen.queryByRole("listbox")).toBeNull()
    expect(draft).toHaveFocus()

    fireEvent.keyDown(draft, { key: "Escape" })
    expect(draft).not.toHaveFocus()
    expect(draft.value).toBe("@a")
  })
})

describe("ChatInput picker", () => {
  afterEach(cleanup)

  const options = ["./scripts/setup.sh", "./README.md", "./notes/readme-old.md"]

  function renderPicker() {
    const onSend = vi.fn()
    const picker = {
      label: "Files changed by this session",
      items: (query: string) => options.filter((option) => option.toLowerCase().includes(query)).map((option) => ({ key: option, label: option })),
      onPick: vi.fn(),
      onCancel: vi.fn(),
    }
    render(<ChatInput chatId="c1" onSend={onSend} picker={picker} />)
    return { onSend, picker, draft: screen.getByRole("textbox") as HTMLTextAreaElement }
  }

  it("takes focus, filters by the draft and picks the arrowed row on Enter without sending it", () => {
    const { onSend, picker, draft } = renderPicker()
    expect(draft).toHaveFocus()
    fireEvent.change(draft, { target: { value: "readme" } })
    expect(screen.getAllByRole("option").map((option) => option.textContent)).toEqual(["./README.md", "./notes/readme-old.md"])
    fireEvent.keyDown(draft, { key: "ArrowDown" })
    fireEvent.keyDown(draft, { key: "Enter" })
    expect(picker.onPick).toHaveBeenCalledWith("./notes/readme-old.md")
    expect(onSend).not.toHaveBeenCalled()
    expect(draft.value).toBe("")
  })

  it("keeps the menu open without matches, Enter then neither picks nor sends, Escape cancels", () => {
    const { onSend, picker, draft } = renderPicker()
    fireEvent.change(draft, { target: { value: "zzz" } })
    expect(screen.getByRole("listbox", { name: "Files changed by this session" })).toHaveTextContent("No matches")
    fireEvent.keyDown(draft, { key: "Enter" })
    expect(picker.onPick).not.toHaveBeenCalled()
    expect(onSend).not.toHaveBeenCalled()
    fireEvent.keyDown(draft, { key: "Escape" })
    expect(picker.onCancel).toHaveBeenCalledOnce()
    expect(draft.value).toBe("")
  })
})

describe("ChatInput while streaming", () => {
  afterEach(cleanup)

  function renderStreaming(sendWhileStreaming: boolean) {
    const onSend = vi.fn()
    const onCancel = vi.fn()
    render(<ChatInput chatId="c1" onSend={onSend} onCancel={onCancel} isStreaming sendWhileStreaming={sendWhileStreaming} />)
    return { onSend, onCancel, draft: screen.getByRole("textbox") as HTMLTextAreaElement }
  }

  it("blocks sends by default, the button only stops", () => {
    const { onSend, draft } = renderStreaming(false)
    fireEvent.change(draft, { target: { value: "more" } })
    fireEvent.keyDown(draft, { key: "Enter" })
    expect(onSend).not.toHaveBeenCalled()
    expect(screen.getByRole("button", { name: "Stop generating" })).toBeInTheDocument()
  })

  it("with sendWhileStreaming, sends a draft by Enter or button and stops with an empty draft", () => {
    const { onSend, onCancel, draft } = renderStreaming(true)
    fireEvent.click(screen.getByRole("button", { name: "Stop generating" }))
    expect(onCancel).toHaveBeenCalledOnce()
    fireEvent.change(draft, { target: { value: "steer this" } })
    fireEvent.keyDown(draft, { key: "Enter" })
    expect(onSend).toHaveBeenLastCalledWith("steer this", [])
    fireEvent.change(draft, { target: { value: "and this" } })
    fireEvent.click(screen.getByRole("button", { name: "Send message" }))
    expect(onSend).toHaveBeenLastCalledWith("and this", [])
    expect(screen.getByRole("button", { name: "Stop generating" })).toBeInTheDocument()
  })
})

describe("ChatInput draft", () => {
  beforeEach(() => {
    const stored = new Map<string, string>()
    vi.stubGlobal("localStorage", {
      getItem: (key: string) => stored.get(key) ?? null,
      setItem: (key: string, value: string) => { stored.set(key, value) },
      removeItem: (key: string) => { stored.delete(key) },
    })
  })
  afterEach(() => { cleanup(); vi.unstubAllGlobals() })

  const draftOf = () => screen.getByRole("textbox") as HTMLTextAreaElement

  it("keeps unsent text of its conversation across closing, and drops it once sent", () => {
    const onSend = vi.fn()
    const first = render(<ChatInput chatId="c1" draftKey="chat:a" onSend={onSend} />)
    fireEvent.change(draftOf(), { target: { value: "half a thought" } })
    first.unmount()

    render(<ChatInput chatId="c2" draftKey="chat:b" onSend={onSend} />)
    expect(draftOf().value).toBe("")
    cleanup()

    render(<ChatInput chatId="c3" draftKey="chat:a" onSend={onSend} />)
    expect(draftOf().value).toBe("half a thought")
    fireEvent.keyDown(draftOf(), { key: "Enter" })
    expect(onSend).toHaveBeenCalledWith("half a thought", [])
    cleanup()

    render(<ChatInput chatId="c4" draftKey="chat:a" onSend={onSend} />)
    expect(draftOf().value).toBe("")
  })
})

describe("ChatInput drawing", () => {
  // jsdom has no PointerEvent: without it the canvas never sees which button or pointer drew
  beforeEach(() => vi.stubGlobal("PointerEvent", class extends MouseEvent {
    pointerType: string
    constructor(type: string, init: PointerEventInit = {}) { super(type, init); this.pointerType = init.pointerType ?? "" }
  }))
  afterEach(() => { cleanup(); vi.unstubAllGlobals() })

  it("uploads a drawing of a project chat under the chat's project", async () => {
    render(<ChatInput chatId="c1" slug="physics" onSend={() => {}} />)
    fireEvent.click(screen.getByRole("button", { name: "Add attachment" }))
    fireEvent.click(screen.getByText("Drawing"))
    const canvas = document.querySelector("svg.w-full") as SVGSVGElement
    canvas.setPointerCapture = () => {}
    fireEvent.pointerDown(canvas, { button: 0, pointerType: "mouse", clientX: 1, clientY: 1 })
    fireEvent.pointerMove(canvas, { pointerType: "mouse", clientX: 5, clientY: 5 })
    fireEvent.pointerUp(canvas, { pointerType: "mouse" })
    await act(async () => { fireEvent.click(screen.getByRole("button", { name: /Confirm/ })) })

    expect(uploadFile).toHaveBeenCalledWith("c1", expect.anything(), "physics")
  })
})

describe("ChatInput PDF text choice", () => {
  afterEach(cleanup)

  async function uploadPdf(onSend: (text: string, attachments: Attachment[]) => void) {
    vi.mocked(uploadFile).mockResolvedValueOnce({ name: "paper.pdf", mime: "application/pdf", url: "/paper.pdf", active: true, textLayer: "Own text" })
    const ref = { current: null as import("../ChatInput").ChatInputHandle | null }
    render(<ChatInput ref={ref} chatId="c1" onSend={onSend} />)
    await act(async () => { ref.current?.addFiles([new File(["%PDF"], "paper.pdf", { type: "application/pdf" })]) })
    return screen.getByRole("group", { name: "paper.pdf has text. Extract it with MinerU instead?" })
  }

  it("Escape sends the PDF's own text, no MinerU parse", async () => {
    const onSend = vi.fn()
    await uploadPdf(onSend)
    const draft = screen.getByRole("textbox")
    fireEvent.keyDown(draft, { key: "Enter" })
    expect(onSend).not.toHaveBeenCalled()

    fireEvent.keyDown(window, { key: "Escape" })
    fireEvent.keyDown(draft, { key: "Enter" })

    expect(screen.queryByRole("group")).toBeNull()
    const [sent] = onSend.mock.calls[0][1] as Attachment[]
    expect(sent.parsedMd).toBe("Own text")
    expect(sent.textLayer).toBeUndefined()
  })

  it("Enter leaves the PDF to MinerU", async () => {
    const onSend = vi.fn()
    const question = await uploadPdf(onSend)

    fireEvent.keyDown(question, { key: "Enter" })
    fireEvent.keyDown(screen.getByRole("textbox"), { key: "Enter" })

    const [sent] = onSend.mock.calls[0][1] as Attachment[]
    expect(sent.parsedMd).toBeUndefined()
    expect(sent.textLayer).toBeUndefined()
  })
})
