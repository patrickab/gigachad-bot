import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"

const { impls } = vi.hoisted(() => ({
  impls: {
    listProjectDocuments: async () => [] as any[],
    listProjectVaultDocuments: async () => [] as any[],
    listAllDocuments: async () => [] as any[],
    addDocument: async () => undefined,
    removeDocument: async () => undefined,
    writeDocument: async () => undefined,
    uploadDocument: async () => undefined,
    uploadFile: async () => ({ name: "u.jpg", mime: "image/jpeg", url: "u", active: true }),
    attachDocument: async () => ({ name: "d", mime: "text/plain", url: "u", active: true }),
    attachFileVaultFile: async () => ({ name: "v", mime: "text/plain", url: "u", active: true }),
    loadFileViewerText: async () => "",
  } as Record<string, (...a: any[]) => any>,
}))

vi.mock("@/lib/api", () => ({
  __setImpl: (k: string, fn: any) => { impls[k] = fn },
  __resetImpls: () => {
    impls.listProjectDocuments = async () => []
    impls.listProjectVaultDocuments = async () => []
    impls.listAllDocuments = async () => []
    impls.addDocument = async () => undefined
    impls.removeDocument = async () => undefined
    impls.writeDocument = async () => undefined
    impls.uploadDocument = async () => undefined
    impls.uploadFile = async () => ({ name: "u.jpg", mime: "image/jpeg", url: "u", active: true })
    impls.attachDocument = async () => ({ name: "d", mime: "text/plain", url: "u", active: true })
    impls.attachFileVaultFile = async () => ({ name: "v", mime: "text/plain", url: "u", active: true })
    impls.loadFileViewerText = async () => ""
  },
  listProjectDocuments: (...a: any[]) => impls.listProjectDocuments(...a),
  listProjectVaultDocuments: (...a: any[]) => impls.listProjectVaultDocuments(...a),
  listAllDocuments: (...a: any[]) => impls.listAllDocuments(...a),
  addDocument: (...a: any[]) => impls.addDocument(...a),
  removeDocument: (...a: any[]) => impls.removeDocument(...a),
  writeDocument: (...a: any[]) => impls.writeDocument(...a),
  uploadDocument: (...a: any[]) => impls.uploadDocument(...a),
  uploadFile: (...a: any[]) => impls.uploadFile(...a),
  attachDocument: (...a: any[]) => impls.attachDocument(...a),
  attachFileVaultFile: (...a: any[]) => impls.attachFileVaultFile(...a),
  loadFileViewerText: (...a: any[]) => impls.loadFileViewerText(...a),
}))

vi.mock("@/lib/drawing", () => ({
  renderCanvasToJpeg: vi.fn(async () => new Blob([new Uint8Array([0])], { type: "image/jpeg" })),
}))
vi.mock("@/lib/attachments", () => ({
  buildHiddenContent: (atts: any[]) => (atts.map((a) => a.content || "").filter(Boolean).join("\n") || null),
}))

import * as apiMock from "@/lib/api"
import { renderHook, act, waitFor } from "@testing-library/react"
import type { ChatInputHandle } from "@/components/ChatInput"
import { useProjectDocuments } from "@/hooks/useProjectDocuments"
import type { ProjectDocument } from "@/lib/types"
import { renderCanvasToJpeg } from "@/lib/drawing"

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
  describe("mergedDocuments / vaultDocPaths dedup", () => {
    it("lists project docs on mount and merges vault docs (deduping by path)", async () => {
      const projectDocs: ProjectDocument[] = [
        { path: "/lib/a.pdf", name: "a.pdf", mime: "application/pdf" },
      ]
      const vaultDocs: ProjectDocument[] = [
        { path: "/lib/a.pdf", name: "a.pdf", mime: "application/pdf" },  // dup — should drop
        { path: "/vault/b.txt", name: "b.txt", mime: "text/plain" },     // vault-only → surfaced
      ]
      ;(apiMock as any).__setImpl("listProjectDocuments", async () => projectDocs)
      ;(apiMock as any).__setImpl("listProjectVaultDocuments", async () => vaultDocs)

      const { result } = renderHook(() => useProjectDocuments(baseProps()))

      await waitFor(() => expect(result.current.projectDocuments).toHaveLength(1))
      expect(result.current.mergedDocuments).toHaveLength(2)
      // vaultDocPaths only contains the vault-only path (not the dup).
      expect(result.current.vaultDocPaths.has("/vault/b.txt")).toBe(true)
      expect(result.current.vaultDocPaths.has("/lib/a.pdf")).toBe(false)
    })

    it("clears document lists when activeProject becomes null", async () => {
      let activeProject: string | null = "proj"
      const { result, rerender } = renderHook(() =>
        useProjectDocuments(baseProps({ activeProject })),
      )
      await waitFor(() => expect(result.current.projectDocuments).toEqual([]))
      activeProject = null
      rerender()
      await waitFor(() => expect(result.current.projectDocuments).toEqual([]))
    })
  })

  describe("handleDocumentSelect", () => {
    it("attaches a vault doc via attachFileVaultFile when path is vault-surfaced", async () => {
      const vaultDocs: ProjectDocument[] = [{ path: "/vault/v.txt", name: "v.txt", mime: "text/plain" }]
      ;(apiMock as any).__setImpl("listProjectVaultDocuments", async () => vaultDocs)
      const attachVault = vi.fn(async () => ({ name: "v.txt", mime: "text/plain", url: "u", active: true }))
      ;(apiMock as any).__setImpl("attachFileVaultFile", attachVault)
      const addAttachment = vi.fn()
      const chatInputRef = makeChatInputRef({ addAttachment })

      const { result } = renderHook(() =>
        useProjectDocuments(baseProps({ chatInputRef })),
      )
      await waitFor(() => expect(result.current.vaultDocPaths.size).toBe(1))

      await act(async () => { await result.current.handleDocumentSelect("/vault/v.txt") })

      expect(attachVault).toHaveBeenCalledWith("/vault/v.txt")
      expect(addAttachment).toHaveBeenCalledOnce()
      expect(result.current.documentOpen).toBe(false)
    })

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

    it("renders a .canvas file to jpeg and uploads it (prefers live canvas content)", async () => {
      const liveCanvasRef = makeLiveCanvasRef("/lib/x.canvas", '{"strokes":[1],"texts":[]}')
      const addAttachment = vi.fn()
      const chatInputRef = makeChatInputRef({ addAttachment })
      const uploadFile = vi.fn(async () => ({ name: "x.jpg", mime: "image/jpeg", url: "u", active: true }))
      ;(apiMock as any).__setImpl("uploadFile", uploadFile)

      const { result } = renderHook(() =>
        useProjectDocuments(baseProps({ chatInputRef, liveCanvasRef })),
      )

      await act(async () => { await result.current.handleDocumentSelect("/lib/x.canvas") })

      // live canvas content was used, not loadFileViewerText.
      expect(renderCanvasToJpeg).toHaveBeenCalledWith([1], 20, [], [])
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

      // Functional update called twice: n => n+1 then n => n-1.
      expect(setExtracting).toHaveBeenCalledTimes(2)
    })
  })

  describe("CRUD handlers", () => {
    it("handleCreateDocument writes + closes panel + refreshes", async () => {
      const writeSpy = vi.fn(async () => undefined)
      ;(apiMock as any).__setImpl("writeDocument", writeSpy)
      const { result } = renderHook(() => useProjectDocuments(baseProps()))

      await act(async () => { await result.current.handleCreateDocument("NewDoc") })

      expect(writeSpy).toHaveBeenCalledWith("proj", "NewDoc")
      expect(result.current.createDocOpen).toBe(false)
    })

    it("handleCreateDocument no-ops without a project", async () => {
      const writeSpy = vi.fn(async () => undefined)
      ;(apiMock as any).__setImpl("writeDocument", writeSpy)
      const { result } = renderHook(() =>
        useProjectDocuments(baseProps({ activeProject: null })),
      )
      await act(async () => { await result.current.handleCreateDocument("x") })
      expect(writeSpy).not.toHaveBeenCalled()
    })

    it("handleDeleteDocument removes from the active project", async () => {
      const removeSpy = vi.fn(async () => undefined)
      ;(apiMock as any).__setImpl("removeDocument", removeSpy)
      const { result } = renderHook(() => useProjectDocuments(baseProps()))
      await act(async () => { await result.current.handleDeleteDocument("/lib/a.pdf") })
      expect(removeSpy).toHaveBeenCalledWith("proj", "/lib/a.pdf")
    })

    it("handleDeleteDocument no-ops without a project", async () => {
      const removeSpy = vi.fn(async () => undefined)
      ;(apiMock as any).__setImpl("removeDocument", removeSpy)
      const { result } = renderHook(() =>
        useProjectDocuments(baseProps({ activeProject: null })),
      )
      await act(async () => { await result.current.handleDeleteDocument("/x") })
      expect(removeSpy).not.toHaveBeenCalled()
    })

    it("handleDocumentUpload uploads every file and refreshes", async () => {
      const uploadSpy = vi.fn(async () => undefined)
      ;(apiMock as any).__setImpl("uploadDocument", uploadSpy)
      const files = [new File(["a"], "a.txt"), new File(["b"], "b.txt")]
      const { result } = renderHook(() => useProjectDocuments(baseProps()))
      await act(async () => { await result.current.handleDocumentUpload(files) })
      expect(uploadSpy).toHaveBeenCalledTimes(2)
      expect(uploadSpy).toHaveBeenCalledWith("proj", files[0])
    })

    it("handleAddDocToProject adds the path to the active project", async () => {
      const addSpy = vi.fn(async () => undefined)
      ;(apiMock as any).__setImpl("addDocument", addSpy)
      const { result } = renderHook(() => useProjectDocuments(baseProps()))
      await act(async () => { await result.current.handleAddDocToProject("/x/y.pdf") })
      expect(addSpy).toHaveBeenCalledWith("proj", "/x/y.pdf")
    })
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

  describe("openDocuments", () => {
    it("refreshes and opens the document picker", async () => {
      const { result } = renderHook(() => useProjectDocuments(baseProps()))
      act(() => result.current.openDocuments())
      expect(result.current.documentOpen).toBe(true)
    })
  })

  describe("Alt+X keyboard shortcut", () => {
    it("toggles the document picker when active in chat mode", async () => {
      const { result } = renderHook(() => useProjectDocuments(baseProps()))
      await waitFor(() => expect(result.current.documentOpen).toBe(false))
      await act(() => {
        window.dispatchEvent(new KeyboardEvent("keydown", { altKey: true, key: "x" }))
      })
      expect(result.current.documentOpen).toBe(true)
      await act(() => {
        window.dispatchEvent(new KeyboardEvent("keydown", { altKey: true, key: "X" }))
      })
      expect(result.current.documentOpen).toBe(false)
    })

    it("does not toggle when in canvas mode", async () => {
      const { result } = renderHook(() => useProjectDocuments(baseProps({ appMode: "canvas" })))
      await act(() => {
        window.dispatchEvent(new KeyboardEvent("keydown", { altKey: true, key: "x" }))
      })
      expect(result.current.documentOpen).toBe(false)
    })

    it("does not toggle when not active", async () => {
      const { result } = renderHook(() => useProjectDocuments(baseProps({ isActive: false })))
      await act(() => {
        window.dispatchEvent(new KeyboardEvent("keydown", { altKey: true, key: "x" }))
      })
      expect(result.current.documentOpen).toBe(false)
    })
  })
})
