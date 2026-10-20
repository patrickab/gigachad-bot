import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"

type MockImpl = (...args: unknown[]) => unknown

const { impls } = vi.hoisted(() => ({
  impls: {
    listProjectDocuments: async () => [],
    listAllDocuments: async () => [],
    addDocument: async () => undefined,
    uploadFile: async () => ({ name: "u.jpg", mime: "image/jpeg", url: "u", active: true }),
    attachReference: async () => ({ name: "d", mime: "text/plain", url: "u", active: true }),
    loadFileViewerText: async () => "",
  } as Record<string, MockImpl>,
}))

vi.mock("@/lib/backend/api", () => ({
  __setImpl: (k: string, fn: (...a: never[]) => unknown) => { impls[k] = fn as MockImpl },
  __resetImpls: () => {
    impls.listProjectDocuments = async () => []
    impls.listAllDocuments = async () => []
    impls.addDocument = async () => undefined
    impls.uploadFile = async () => ({ name: "u.jpg", mime: "image/jpeg", url: "u", active: true })
    impls.attachReference = async () => ({ name: "d", mime: "text/plain", url: "u", active: true })
    impls.loadFileViewerText = async () => ""
  },
  listProjectDocuments: (...a: unknown[]) => impls.listProjectDocuments(...a),
  listAllDocuments: (...a: unknown[]) => impls.listAllDocuments(...a),
  addDocument: (...a: unknown[]) => impls.addDocument(...a),
  uploadFile: (...a: unknown[]) => impls.uploadFile(...a),
  attachReference: (...a: unknown[]) => impls.attachReference(...a),
  loadFileViewerText: (...a: unknown[]) => impls.loadFileViewerText(...a),
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

/** Controls the mocked api module exposes beyond the real exports. */
const mockControls = apiMock as unknown as {
  __setImpl: (name: string, impl: (...args: never[]) => unknown) => void
  __resetImpls: () => void
}
import { renderHook, act, waitFor } from "@testing-library/react"
import type { ChatInputHandle } from "@/features/chat/composer/ChatInput"
import { useProjectDocuments } from "@/features/chat/useProjectDocuments"
import { renderCanvasToJpeg } from "@/lib/drawing"
import { fileViewerRawUrl } from "@/lib/backend/urls"

function makeChatInputRef(handle: Partial<ChatInputHandle> = {}) {
  return { current: { addAttachment: vi.fn(), ...handle } as unknown as ChatInputHandle }
}

beforeEach(() => mockControls.__resetImpls())
afterEach(() => { vi.useRealTimers(); vi.clearAllMocks() })

const baseProps = (over: Partial<Parameters<typeof useProjectDocuments>[0]> = {}) => ({
  isActive: true,
  appMode: "chat" as const,
  projectSlug: "proj",
  chatId: "c1",
  chatInputRef: makeChatInputRef(),
  setExtracting: vi.fn(),
  ...over,
})

describe("useProjectDocuments", () => {
  describe("handleDocumentSelect", () => {
    it("attaches a project document as a live reference", async () => {
      const attachDoc = vi.fn(async () => ({ name: "doc.md", mime: "text/markdown", url: "u", active: true }))
      mockControls.__setImpl("attachReference", attachDoc)
      const addAttachment = vi.fn()
      const chatInputRef = makeChatInputRef({ addAttachment })

      const { result } = renderHook(() =>
        useProjectDocuments(baseProps({ chatInputRef })),
      )

      await act(async () => { await result.current.handleDocumentSelect("/lib/doc.md") })

      expect(attachDoc).toHaveBeenCalledWith("/lib/doc.md")
      expect(addAttachment).toHaveBeenCalledOnce()
    })

    it("renders image attachments but excludes PDFs from a canvas JPEG", async () => {
      mockControls.__setImpl("loadFileViewerText", async () => JSON.stringify({
        version: 1,
        strokes: [],
        texts: [],
        frames: [],
        attachments: [
          { kind: "document", path: "/lib/plot.png", x: 10, y: 20, width: 30, height: 60 },
          { kind: "pdf", path: "/lib/source.pdf", x: 40, y: 50, width: 70 },
        ],
      }))
      const { result } = renderHook(() => useProjectDocuments(baseProps()))

      await act(async () => { await result.current.handleDocumentSelect("/lib/x.canvas") })

      expect(renderCanvasToJpeg).toHaveBeenCalledWith([], 20, [
        { url: fileViewerRawUrl("/lib/plot.png"), x: 10, y: 20, width: 30, aspect: 2 },
      ], [])
    })

    it("renders a saved .canvas file to jpeg and uploads it", async () => {
      const stroke = { id: "s1", points: [[0, 0]], color: "#000", width: 2 }
      mockControls.__setImpl("loadFileViewerText", async () => JSON.stringify({ version: 1, strokes: [stroke], texts: [] }))
      const addAttachment = vi.fn()
      const chatInputRef = makeChatInputRef({ addAttachment })
      const uploadFile = vi.fn(async () => ({ name: "x.jpg", mime: "image/jpeg", url: "u", active: true }))
      mockControls.__setImpl("uploadFile", uploadFile)

      const { result } = renderHook(() =>
        useProjectDocuments(baseProps({ chatInputRef })),
      )

      await act(async () => { await result.current.handleDocumentSelect("/lib/x.canvas") })

      expect(renderCanvasToJpeg).toHaveBeenCalledWith([stroke], 20, [], [])
      expect(uploadFile).toHaveBeenCalledWith("c1", expect.any(File), "proj", true)
      expect(addAttachment).toHaveBeenCalledOnce()
    })

    it("skips an empty canvas (no strokes, no texts) silently", async () => {
      mockControls.__setImpl("loadFileViewerText", async () => "{}")
      const addAttachment = vi.fn()
      const chatInputRef = makeChatInputRef({ addAttachment })

      const { result } = renderHook(() =>
        useProjectDocuments(baseProps({ chatInputRef })),
      )

      await act(async () => { await result.current.handleDocumentSelect("/lib/empty.canvas") })

      expect(addAttachment).not.toHaveBeenCalled()
    })

    it("restores extracting counter even if attach throws", async () => {
      mockControls.__setImpl("attachReference", async () => { throw new Error("boom") })
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
    mockControls.__setImpl("listProjectDocuments", async () => docs)
    mockControls.__setImpl("loadFileViewerText", async (path: string) => path.includes("backend") ? "backend = system" : "frontend = system")
    const upload = vi.fn(async (_chat: string, file: File) => ({ name: file.name, mime: file.type, url: "u", active: true }))
    mockControls.__setImpl("uploadFile", upload)
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
})
