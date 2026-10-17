import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"

const { impls } = vi.hoisted(() => ({
  impls: {
    listProjectDocuments: async () => [] as any[],
    listAllDocuments: async () => [] as any[],
    addDocument: async () => undefined,
    removeDocument: async () => undefined,
    writeDocument: async () => undefined,
    uploadDocument: async () => undefined,
    uploadFile: async () => ({ name: "u.jpg", mime: "image/jpeg", url: "u", active: true }),
    attachDocument: async () => ({ name: "d", mime: "text/plain", url: "u", active: true }),
    loadFileViewerText: async () => "",
    renameDocument: async () => ({ path: "project/proj/document/renamed.md", name: "renamed.md", mime: "text/markdown" }),
  } as Record<string, (...a: any[]) => any>,
}))

vi.mock("@/lib/backend/api", () => ({
  __setImpl: (k: string, fn: any) => { impls[k] = fn },
  __resetImpls: () => {
    impls.listProjectDocuments = async () => []
    impls.listAllDocuments = async () => []
    impls.addDocument = async () => undefined
    impls.removeDocument = async () => undefined
    impls.writeDocument = async () => undefined
    impls.uploadDocument = async () => undefined
    impls.uploadFile = async () => ({ name: "u.jpg", mime: "image/jpeg", url: "u", active: true })
    impls.attachDocument = async () => ({ name: "d", mime: "text/plain", url: "u", active: true })
    impls.loadFileViewerText = async () => ""
    impls.renameDocument = async () => ({ path: "project/proj/document/renamed.md", name: "renamed.md", mime: "text/markdown" })
  },
  listProjectDocuments: (...a: any[]) => impls.listProjectDocuments(...a),
  listAllDocuments: (...a: any[]) => impls.listAllDocuments(...a),
  addDocument: (...a: any[]) => impls.addDocument(...a),
  removeDocument: (...a: any[]) => impls.removeDocument(...a),
  writeDocument: (...a: any[]) => impls.writeDocument(...a),
  uploadDocument: (...a: any[]) => impls.uploadDocument(...a),
  uploadFile: (...a: any[]) => impls.uploadFile(...a),
  attachDocument: (...a: any[]) => impls.attachDocument(...a),
  loadFileViewerText: (...a: any[]) => impls.loadFileViewerText(...a),
  renameDocument: (...a: any[]) => impls.renameDocument(...a),
}))

vi.mock("@/lib/drawing", () => ({
  renderCanvasToJpeg: vi.fn(async () => new Blob([new Uint8Array([0])], { type: "image/jpeg" })),
}))
vi.mock("@/features/chat/attachments", () => ({
  withAttachments: (msg: object, atts: { content?: string }[]) => ({
    ...msg,
    attachments: atts,
    hiddenContent: atts.map((a) => a.content || "").filter(Boolean).join("\n") || undefined,
  }),
}))

import * as apiMock from "@/lib/backend/api"
import { renderHook as testingRenderHook, act, waitFor, screen, fireEvent } from "@testing-library/react"
import type { ChatInputHandle } from "@/features/chat/composer/ChatInput"
import { useProjectDocuments } from "@/features/chat/useProjectDocuments"
import { UndoDeleteProvider } from "@/contexts/UndoDeleteContext"
import { renderCanvasToJpeg } from "@/lib/drawing"
import { fileViewerRawUrl } from "@/lib/backend/urls"

const renderHook = <T,>(callback: () => T) => testingRenderHook(callback, { wrapper: UndoDeleteProvider })
function makeChatInputRef(handle: Partial<ChatInputHandle> = {}) {
  return { current: { addAttachment: vi.fn(), ...handle } as unknown as ChatInputHandle }
}
function makeLiveCanvasRef(path: string | null = null, content = "") {
  return { current: path ? { path, content } : null }
}

beforeEach(() => (apiMock as any).__resetImpls())
afterEach(() => { vi.useRealTimers(); vi.clearAllMocks() })

const baseProps = (over: any = {}) => ({
  isActive: true,
  appMode: "chat" as const,
  activeProject: "proj",
  chatId: "c1",
  chatInputRef: makeChatInputRef(),
  liveCanvasRef: makeLiveCanvasRef(),
  setExtracting: vi.fn(),
  setMessages: vi.fn(),
  ...over,
})

describe("useProjectDocuments", () => {
  describe("handleDocumentSelect", () => {
    it("attaches a regular project document via attachDocument", async () => {
      const attachDoc = vi.fn(async () => ({ name: "doc.md", mime: "text/markdown", url: "u", active: true }))
      ;(apiMock as any).__setImpl("attachDocument", attachDoc)
      const addAttachment = vi.fn()
      const chatInputRef = makeChatInputRef({ addAttachment })

      const { result } = renderHook(() =>
        useProjectDocuments(baseProps({ chatInputRef })),
      )

      await act(async () => { await result.current.handleDocumentSelect("/lib/doc.md") })

      expect(attachDoc).toHaveBeenCalledWith("c1", "/lib/doc.md", "proj")
      expect(addAttachment).toHaveBeenCalledOnce()
    })

    it("renders image attachments but excludes PDFs from a canvas JPEG", async () => {
      const liveCanvasRef = makeLiveCanvasRef("/lib/x.canvas", JSON.stringify({
        version: 1,
        strokes: [],
        texts: [],
        frames: [],
        attachments: [
          { kind: "document", path: "/lib/plot.png", x: 10, y: 20, width: 30, height: 60 },
          { kind: "pdf", path: "/lib/source.pdf", x: 40, y: 50, width: 70 },
        ],
      }))
      const { result } = renderHook(() => useProjectDocuments(baseProps({ liveCanvasRef })))

      await act(async () => { await result.current.handleDocumentSelect("/lib/x.canvas") })

      expect(renderCanvasToJpeg).toHaveBeenCalledWith([], 20, [
        { url: fileViewerRawUrl("/lib/plot.png"), x: 10, y: 20, width: 30, aspect: 2 },
      ], [])
    })

    it("renders a .canvas file to jpeg and uploads it (prefers live canvas content)", async () => {
      const stroke = { id: "s1", points: [[0, 0]], color: "#000", width: 2 }
      const liveCanvasRef = makeLiveCanvasRef("/lib/x.canvas", JSON.stringify({ version: 1, strokes: [stroke], texts: [] }))
      const addAttachment = vi.fn()
      const chatInputRef = makeChatInputRef({ addAttachment })
      const uploadFile = vi.fn(async () => ({ name: "x.jpg", mime: "image/jpeg", url: "u", active: true }))
      ;(apiMock as any).__setImpl("uploadFile", uploadFile)

      const { result } = renderHook(() =>
        useProjectDocuments(baseProps({ chatInputRef, liveCanvasRef })),
      )

      await act(async () => { await result.current.handleDocumentSelect("/lib/x.canvas") })

      expect(renderCanvasToJpeg).toHaveBeenCalledWith([stroke], 20, [], [])
      expect(uploadFile).toHaveBeenCalledWith("c1", expect.any(File), "proj", true)
      expect(addAttachment).toHaveBeenCalledOnce()
    })

    it("skips an empty canvas (no strokes, no texts) silently", async () => {
      const liveCanvasRef = makeLiveCanvasRef("/lib/empty.canvas", "{}")
      const addAttachment = vi.fn()
      const chatInputRef = makeChatInputRef({ addAttachment })

      const { result } = renderHook(() =>
        useProjectDocuments(baseProps({ chatInputRef, liveCanvasRef })),
      )

      await act(async () => { await result.current.handleDocumentSelect("/lib/empty.canvas") })

      expect(addAttachment).not.toHaveBeenCalled()
    })

    it("restores extracting counter even if attach throws", async () => {
      ;(apiMock as any).__setImpl("attachDocument", async () => { throw new Error("boom") })
      const setExtracting = vi.fn()
      const chatInputRef = makeChatInputRef({ addAttachment: vi.fn() })

      const { result } = renderHook(() =>
        useProjectDocuments(baseProps({ chatInputRef, setExtracting })),
      )

      await act(async () => { await result.current.handleDocumentSelect("/lib/doc.md") })

      expect(setExtracting).toHaveBeenCalledTimes(2)
    })
  })

  it("attaches all C4 sources as one named chat context with file boundaries", async () => {
    const docs = [
      { path: "graph/proj/backend/backend.c4", name: "backend.c4", mime: "text/plain" },
      { path: "graph/proj/frontend/frontend.c4", name: "frontend.c4", mime: "text/plain" },
    ]
    ;(apiMock as any).__setImpl("listProjectDocuments", async () => docs)
    ;(apiMock as any).__setImpl("loadFileViewerText", async (path: string) => path.includes("backend") ? "backend = system" : "frontend = system")
    const upload = vi.fn(async (_chat: string, file: File) => ({ name: file.name, mime: file.type, url: "u", active: true }))
    ;(apiMock as any).__setImpl("uploadFile", upload)
    const addAttachment = vi.fn()
    const { result } = renderHook(() => useProjectDocuments(baseProps({ chatInputRef: makeChatInputRef({ addAttachment }) })))
    await waitFor(() => expect(result.current.projectDocuments).toHaveLength(2))
    await act(async () => { await result.current.handleAttachArchitecture!() })
    expect(upload).toHaveBeenCalledOnce()
    const attachment = addAttachment.mock.calls[0][0]
    expect(attachment.content).toContain("## backend/backend.c4\n\n```likec4\nbackend = system")
    expect(attachment.content).toContain("## frontend/frontend.c4\n\n```likec4\nfrontend = system")
    expect(attachment.active).toBe(true)
  })

  it("hides a document until Undo restores it without deleting the source", async () => {
    const path = "project/proj/document/spec.md"
    const remove = vi.fn(async () => undefined)
    ;(apiMock as any).__setImpl("listProjectDocuments", async () => [{ path, name: "spec.md", mime: "text/markdown" }])
    ;(apiMock as any).__setImpl("removeDocument", remove)
    const { result } = renderHook(() => useProjectDocuments(baseProps()))
    await waitFor(() => expect(result.current.visibleDocuments).toHaveLength(1))
    act(() => result.current.handleDeleteDocument(path))
    expect(result.current.visibleDocuments).toHaveLength(0)
    fireEvent.click(screen.getByRole("button", { name: "Undo" }))
    expect(result.current.visibleDocuments).toHaveLength(1)
    expect(remove).not.toHaveBeenCalled()
  })

  describe("handleDocumentSaved", () => {
    it("refreshes and patches messages whose attachments match the saved filename", async () => {
      const setMessages = vi.fn()
      const { result } = renderHook(() =>
        useProjectDocuments(baseProps({ setMessages })),
      )

      act(() => {
        result.current.handleDocumentSaved("doc.md", "new content")
      })

      expect(setMessages).toHaveBeenCalledOnce()
      const updater = setMessages.mock.calls[0][0] as (prev: any[]) => any[]
      const prev = [
        { role: "user", content: "q", attachments: [{ name: "doc.md", content: "old", active: true }] },
        { role: "assistant", content: "a" },
      ]
      const out = updater(prev)
      expect(out[0].attachments[0].content).toBe("new content")
      expect(out[0].hiddenContent).toContain("new content")
    })

    it("refreshes but does not patch messages when no filename provided", () => {
      const setMessages = vi.fn()
      const { result } = renderHook(() =>
        useProjectDocuments(baseProps({ setMessages })),
      )
      act(() => { result.current.handleDocumentSaved() })
      expect(setMessages).not.toHaveBeenCalled()
    })
  })

})
