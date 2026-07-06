import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"

const { impls } = vi.hoisted(() => ({
  impls: {
    saveChatHistory: async () => undefined,
    saveProjectTab: async () => undefined,
    buildHistoryFile: (filename: string, slug: string | null) =>
      slug ? `${slug}/${filename}` : filename,
    parseHistoryFile: (historyFile: string) => {
      const parts = historyFile.split("/")
      if (parts.length > 1) return { slug: parts[0], filename: parts.slice(1).join("/") }
      return { slug: null, filename: historyFile }
    },
    generateMindmap: async () => "# mindmap\n- a\n",
  } as Record<string, (...a: any[]) => any>,
}))

vi.mock("@/lib/api", () => ({
  __setImpl: (k: string, fn: any) => { impls[k] = fn },
  __resetImpls: () => {
    impls.saveChatHistory = async () => undefined
    impls.saveProjectTab = async () => undefined
    impls.buildHistoryFile = (f: string, s: string | null) => (s ? `${s}/${f}` : f)
    impls.parseHistoryFile = (h: string) => {
      const p = h.split("/")
      if (p.length > 1) return { slug: p[0], filename: p.slice(1).join("/") }
      return { slug: null, filename: h }
    }
    impls.generateMindmap = async () => "# mindmap\n- a\n"
  },
  saveChatHistory: (...a: any[]) => impls.saveChatHistory(...a),
  saveProjectTab: (...a: any[]) => impls.saveProjectTab(...a),
  buildHistoryFile: (...a: any[]) => impls.buildHistoryFile(...a),
  parseHistoryFile: (...a: any[]) => impls.parseHistoryFile(...a),
  generateMindmap: (...a: any[]) => impls.generateMindmap(...a),
}))

vi.mock("@/hooks/useStudyHandler", () => ({
  updateLastMsg: (
    setMessages: React.Dispatch<React.SetStateAction<any[]>>,
    updater: (m: any) => any,
  ) => setMessages((prev: any[]) => {
    const copy = [...prev]
    const last = copy[copy.length - 1]
    if (last?.role === "assistant") copy[copy.length - 1] = updater(last)
    return copy
  }),
}))

import * as apiMock from "@/lib/api"
import { useChatModals } from "@/hooks/useChatModals"
import { renderHook, act, waitFor } from "@testing-library/react"
import type { Tab } from "@/components/TabManager"

function makeTab(over: Partial<Tab> = {}): Tab {
  return {
    id: "t1",
    name: "Tab",
    chatId: "chat-1",
    historyFile: null,
    title: null,
    config: {} as any,
    appMode: "chat",
    ...over,
  }
}

beforeEach(() => (apiMock as any).__resetImpls())
afterEach(() => vi.useRealTimers())

describe("useChatModals", () => {
  describe("isTitled", () => {
    it("is false when the tab has no historyFile", () => {
      const { result } = renderHook(() =>
        useChatModals({
          tab: makeTab(),
          activeProject: null,
          messages: [],
          chatId: "c",
          hasUsage: undefined,
          selectedModel: "m",
          refreshAll: async () => {},
          onHistoryFileChanged: vi.fn(),
          setMessages: vi.fn(),
        }),
      )
      expect(result.current.isTitled).toBe(false)
    })

    it("is true when the tab has a historyFile", () => {
      const { result } = renderHook(() =>
        useChatModals({
          tab: makeTab({ historyFile: "foo.json" }),
          activeProject: null,
          messages: [],
          chatId: "c",
          hasUsage: undefined,
          selectedModel: "m",
          refreshAll: async () => {},
          onHistoryFileChanged: vi.fn(),
          setMessages: vi.fn(),
        }),
      )
      expect(result.current.isTitled).toBe(true)
    })
  })

  describe("handleQuickSave", () => {
    it("no-ops when the tab is untitled", async () => {
      const saveChat = vi.fn(async () => undefined)
      const saveTab = vi.fn(async () => undefined)
      ;(apiMock as any).__setImpl("saveChatHistory", saveChat)
      ;(apiMock as any).__setImpl("saveProjectTab", saveTab)
      const refreshAll = vi.fn(async () => {})

      const { result } = renderHook(() =>
        useChatModals({
          tab: makeTab(),
          activeProject: null,
          messages: [],
          chatId: "c",
          hasUsage: undefined,
          selectedModel: "m",
          refreshAll,
          onHistoryFileChanged: vi.fn(),
          setMessages: vi.fn(),
        }),
      )

      await act(async () => { await result.current.handleQuickSave() })

      expect(saveChat).not.toHaveBeenCalled()
      expect(saveTab).not.toHaveBeenCalled()
      expect(refreshAll).not.toHaveBeenCalled()
    })

    it("routes through saveProjectTab when historyFile belongs to the active project", async () => {
      const saveChat = vi.fn(async () => undefined)
      const saveTab = vi.fn(async () => undefined)
      ;(apiMock as any).__setImpl("saveChatHistory", saveChat)
      ;(apiMock as any).__setImpl("saveProjectTab", saveTab)
      const refreshAll = vi.fn(async () => {})
      const usage = { prompt_tokens: 1, completion_tokens: 2, total_tokens: 3 }

      const { result } = renderHook(() =>
        useChatModals({
          tab: makeTab({ historyFile: "proj/tab1.json", name: "n", title: "T" }),
          activeProject: "proj",
          messages: [{ role: "user", content: "hi" }],
          chatId: "chat-1",
          hasUsage: usage,
          selectedModel: "m",
          refreshAll,
          onHistoryFileChanged: vi.fn(),
          setMessages: vi.fn(),
        }),
      )

      await act(async () => { await result.current.handleQuickSave() })

      expect(saveTab).toHaveBeenCalledWith("proj", "tab1.json", [{ role: "user", content: "hi" }], "chat-1", "n", "T", usage)
      expect(saveChat).not.toHaveBeenCalled()
      expect(refreshAll).toHaveBeenCalledOnce()
    })

    it("routes through saveChatHistory when historyFile is a plain path", async () => {
      const saveChat = vi.fn(async () => undefined)
      const saveTab = vi.fn(async () => undefined)
      ;(apiMock as any).__setImpl("saveChatHistory", saveChat)
      ;(apiMock as any).__setImpl("saveProjectTab", saveTab)

      const { result } = renderHook(() =>
        useChatModals({
          tab: makeTab({ historyFile: "free.json", name: null, title: "Free" }),
          activeProject: null,
          messages: [{ role: "user", content: "hi" }],
          chatId: "chat-1",
          hasUsage: undefined,
          selectedModel: "m",
          refreshAll: async () => {},
          onHistoryFileChanged: vi.fn(),
          setMessages: vi.fn(),
        }),
      )

      await act(async () => { await result.current.handleQuickSave() })

      expect(saveChat).toHaveBeenCalledWith("free.json", [{ role: "user", content: "hi" }], "chat-1", "Free", undefined)
      expect(saveTab).not.toHaveBeenCalled()
    })
  })

  describe("handleSaveSubmit", () => {
    it("saves to project and notifies with buildHistoryFile when a project is active", async () => {
      const saveTab = vi.fn(async () => undefined)
      ;(apiMock as any).__setImpl("saveProjectTab", saveTab)
      const onHistoryFileChanged = vi.fn()
      const refreshAll = vi.fn(async () => {})

      const { result } = renderHook(() =>
        useChatModals({
          tab: makeTab({ id: "t1", name: "n" }),
          activeProject: "proj",
          messages: [{ role: "user", content: "x" }],
          chatId: "c1",
          hasUsage: undefined,
          selectedModel: "m",
          refreshAll,
          onHistoryFileChanged,
          setMessages: vi.fn(),
        }),
      )

      await act(async () => { await result.current.handleSaveSubmit("My Title") })

      expect(saveTab).toHaveBeenCalledWith("proj", "My Title.json", expect.any(Array), "c1", "n", "My Title", undefined)
      expect(onHistoryFileChanged).toHaveBeenCalledWith("t1", "proj/My Title.json")
      expect(refreshAll).toHaveBeenCalledOnce()
      expect(result.current.saveModalOpen).toBe(false)
    })

    it("saves standalone and notifies with bare filename when no project", async () => {
      const saveChat = vi.fn(async () => undefined)
      ;(apiMock as any).__setImpl("saveChatHistory", saveChat)
      const onHistoryFileChanged = vi.fn()

      const { result } = renderHook(() =>
        useChatModals({
          tab: makeTab({ id: "t2", name: null }),
          activeProject: null,
          messages: [{ role: "user", content: "x" }],
          chatId: "c2",
          hasUsage: undefined,
          selectedModel: "m",
          refreshAll: async () => {},
          onHistoryFileChanged,
          setMessages: vi.fn(),
        }),
      )

      await act(async () => { await result.current.handleSaveSubmit("Solo") })

      expect(saveChat).toHaveBeenCalledWith("Solo.json", expect.any(Array), "c2", "Solo", undefined)
      expect(onHistoryFileChanged).toHaveBeenCalledWith("t2", "Solo.json")
    })
  })

  describe("handleMindmapSubmit", () => {
    it("appends user + placeholder assistant, then replaces with mindmap on success", async () => {
      const genSpy = vi.fn(async () => "# mindmap result")
      ;(apiMock as any).__setImpl("generateMindmap", genSpy)
      const setMessages = vi.fn()

      const { result } = renderHook(() =>
        useChatModals({
          tab: makeTab(),
          activeProject: null,
          messages: [{ role: "user", content: "q" }],
          chatId: "c",
          hasUsage: undefined,
          selectedModel: "gemini",
          refreshAll: async () => {},
          onHistoryFileChanged: vi.fn(),
          setMessages,
        }),
      )

      await act(async () => {
        await result.current.handleMindmapSubmit("summarize", [])
      })

      // Two setMessages calls: append user+placeholder, then updateLastMsg.
      expect(setMessages).toHaveBeenCalledTimes(2)
      // First call adds the user message and the generating placeholder.
      const firstCall = setMessages.mock.calls[0][0] as (prev: any[]) => any[]
      expect(firstCall([{ role: "user", content: "q" }])).toEqual([
        { role: "user", content: "q" },
        { role: "user", content: "Provide a mindmap. summarize" },
        { role: "assistant", content: "Generating mind map…" },
      ])
      // generateMindmap was invoked with the configured model.
      expect(genSpy).toHaveBeenCalled()
      const args = genSpy.mock.calls[0] as unknown[]
      expect(args[1]).toBe("gemini")
      expect(args[2]).toBe("summarize")
      // Modal closed.
      expect(result.current.mindmapModalOpen).toBe(false)
    })

    it("no-ops when there are no messages", async () => {
      const genSpy = vi.fn(async () => "# x")
      ;(apiMock as any).__setImpl("generateMindmap", genSpy)

      const { result } = renderHook(() =>
        useChatModals({
          tab: makeTab(),
          activeProject: null,
          messages: [],
          chatId: "c",
          hasUsage: undefined,
          selectedModel: "m",
          refreshAll: async () => {},
          onHistoryFileChanged: vi.fn(),
          setMessages: vi.fn(),
        }),
      )

      await act(async () => { await result.current.handleMindmapSubmit("", []) })
      expect(genSpy).not.toHaveBeenCalled()
    })

    it("writes a failure message to the last assistant message when generation throws", async () => {
      ;(apiMock as any).__setImpl("generateMindmap", async () => { throw new Error("boom") })
      const setMessages = vi.fn()

      const { result } = renderHook(() =>
        useChatModals({
          tab: makeTab(),
          activeProject: null,
          messages: [{ role: "user", content: "q" }],
          chatId: "c",
          hasUsage: undefined,
          selectedModel: "m",
          refreshAll: async () => {},
          onHistoryFileChanged: vi.fn(),
          setMessages,
        }),
      )

      await act(async () => { await result.current.handleMindmapSubmit("", []) })

      // Last setMessages call (updateLastMsg) sets the failure message.
      const lastCall = setMessages.mock.calls.at(-1)![0] as (prev: any[]) => any[]
      const out = lastCall([{ role: "user", content: "q" }, { role: "assistant", content: "Generating mind map…" }])
      expect(out.at(-1)!.content).toBe("Mind map generation failed.")
    })
  })

  describe("modal visibility setters", () => {
    it("toggles save modal open state", async () => {
      const { result } = renderHook(() =>
        useChatModals({
          tab: makeTab(),
          activeProject: null,
          messages: [],
          chatId: "c",
          hasUsage: undefined,
          selectedModel: "m",
          refreshAll: async () => {},
          onHistoryFileChanged: vi.fn(),
          setMessages: vi.fn(),
        }),
      )
      expect(result.current.saveModalOpen).toBe(false)
      await act(() => result.current.setSaveModalOpen(true))
      expect(result.current.saveModalOpen).toBe(true)
    })

    it("toggles prompt editor and mindmap modal setters", async () => {
      const { result } = renderHook(() =>
        useChatModals({
          tab: makeTab(),
          activeProject: null,
          messages: [],
          chatId: "c",
          hasUsage: undefined,
          selectedModel: "m",
          refreshAll: async () => {},
          onHistoryFileChanged: vi.fn(),
          setMessages: vi.fn(),
        }),
      )
      await act(() => {
        result.current.setPromptEditorOpen(true)
        result.current.setMindmapModalOpen(true)
        result.current.setMindmapAttachments([{ name: "a", mime: "text/plain", url: "u", active: true } as any])
      })
      expect(result.current.promptEditorOpen).toBe(true)
      expect(result.current.mindmapModalOpen).toBe(true)
      expect(result.current.mindmapAttachments).toHaveLength(1)
    })
  })
})
