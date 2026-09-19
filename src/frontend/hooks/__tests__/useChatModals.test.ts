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

})
