import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"

type MockImplementation = (...args: never[]) => unknown

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
  } as Record<string, MockImplementation>,
}))

vi.mock("@/lib/api", () => ({
  __setImpl: (key: string, fn: MockImplementation) => { impls[key] = fn },
  __resetImpls: () => {
    impls.saveChatHistory = async () => undefined
    impls.saveProjectTab = async () => undefined
    impls.buildHistoryFile = (filename: string, slug: string | null) => (slug ? `${slug}/${filename}` : filename)
    impls.parseHistoryFile = (historyFile: string) => {
      const parts = historyFile.split("/")
      if (parts.length > 1) return { slug: parts[0], filename: parts.slice(1).join("/") }
      return { slug: null, filename: historyFile }
    }
    impls.generateMindmap = async () => "# mindmap\n- a\n"
  },
  saveChatHistory: (...args: never[]) => impls.saveChatHistory(...args),
  saveProjectTab: (...args: never[]) => impls.saveProjectTab(...args),
  buildHistoryFile: (...args: never[]) => impls.buildHistoryFile(...args),
  parseHistoryFile: (...args: never[]) => impls.parseHistoryFile(...args),
  generateMindmap: (...args: never[]) => impls.generateMindmap(...args),
}))

vi.mock("@/lib/utils", () => ({
  cn: (...a: unknown[]) => a.filter(Boolean).join(" "),
  updateLastMsg: (
    setMessages: React.Dispatch<React.SetStateAction<Record<string, unknown>[]>>,
    updater: (m: Record<string, unknown>) => Record<string, unknown>,
  ) => setMessages((prev: Record<string, unknown>[]) => {
    const copy = [...prev]
    const last = copy[copy.length - 1]
    if (last?.role === "assistant") copy[copy.length - 1] = updater(last)
    return copy
  }),
}))


import * as apiMock from "@/lib/api"
import { useChatModals } from "@/hooks/useChatModals"
import { renderHook, act } from "@testing-library/react"
import type { Tab } from "@/components/TabManager"
import type { Message } from "@/lib/types"

interface ApiMockControls {
  __setImpl(key: string, fn: MockImplementation): void
  __resetImpls(): void
}

// The test-only mock controls are injected by vi.mock rather than the production module.
const apiMockControls = apiMock as unknown as ApiMockControls

function makeTab(over: Partial<Tab> = {}): Tab {
  return {
    id: "t1",
    name: "Tab",
    chatId: "chat-1",
    historyFile: null,
    title: null,
    config: {} as Tab["config"],
    appMode: "chat",
    ...over,
  }
}

beforeEach(() => apiMockControls.__resetImpls())
afterEach(() => vi.useRealTimers())

describe("useChatModals", () => {
  describe("handleAutosave", () => {
    it("persists a completed response for a saved chat", async () => {
      const saveSpy = vi.fn(async () => undefined)
      const refreshAll = vi.fn(async () => undefined)
      apiMockControls.__setImpl("saveChatHistory", saveSpy)
      const completedMessages = [
        { role: "user" as const, content: "q" },
        { role: "assistant" as const, content: "a" },
      ]
      const usage = { prompt_tokens: 3, completion_tokens: 5, total_tokens: 8 }

      const { result } = renderHook(() =>
        useChatModals({
          tab: makeTab({ historyFile: "saved.json", title: "Saved" }),
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

      await act(async () => { await result.current.handleAutosave(completedMessages, usage) })

      expect(saveSpy).toHaveBeenCalledWith("saved.json", completedMessages, {
        chatId: "c",
        title: "Saved",
        usage,
      })
      expect(refreshAll).toHaveBeenCalledOnce()
    })

    it("skips an unsaved chat", async () => {
      const saveSpy = vi.fn(async () => undefined)
      apiMockControls.__setImpl("saveChatHistory", saveSpy)
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

      await act(async () => {
        await result.current.handleAutosave([{ role: "assistant", content: "a" }], {
          prompt_tokens: 0,
          completion_tokens: 1,
          total_tokens: 1,
        })
      })

      expect(saveSpy).not.toHaveBeenCalled()
    })
  })

  describe("handleMindmapSubmit", () => {
    it("appends user + placeholder assistant, then replaces with mindmap on success", async () => {
      const genSpy = vi.fn(async () => "# mindmap result")
      apiMockControls.__setImpl("generateMindmap", genSpy)
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
      const firstCall = setMessages.mock.calls[0][0] as (prev: Message[]) => Message[]
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
      apiMockControls.__setImpl("generateMindmap", genSpy)

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
      apiMockControls.__setImpl("generateMindmap", async () => { throw new Error("boom") })
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
      const lastCall = setMessages.mock.calls.at(-1)![0] as (prev: Message[]) => Message[]
      const out = lastCall([{ role: "user", content: "q" }, { role: "assistant", content: "Generating mind map…" }])
      expect(out.at(-1)!.content).toBe("Mind map generation failed.")
    })
  })

})
