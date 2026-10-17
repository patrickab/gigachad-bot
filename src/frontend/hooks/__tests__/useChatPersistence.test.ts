import { describe, it, expect, vi, beforeEach } from "vitest"

type MockImplementation = (...args: never[]) => unknown

const { impls } = vi.hoisted(() => ({
  impls: {
    saveChatHistory: async () => undefined,
    saveProjectTab: async () => undefined,
  } as Record<string, MockImplementation>,
}))

vi.mock("@/lib/api", () => ({
  __setImpl: (key: string, fn: MockImplementation) => { impls[key] = fn },
  __resetImpls: () => {
    impls.saveChatHistory = async () => undefined
    impls.saveProjectTab = async () => undefined
  },
  saveChatHistory: (...args: never[]) => impls.saveChatHistory(...args),
  saveProjectTab: (...args: never[]) => impls.saveProjectTab(...args),
}))

import * as apiMock from "@/lib/api"
import { useChatPersistence } from "@/hooks/useChatPersistence"
import { renderHook, act } from "@testing-library/react"
import type { Tab } from "@/contexts/TabContext"

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

describe("useChatPersistence", () => {
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
        useChatPersistence({
          tab: makeTab({ historyFile: "saved.json", title: "Saved" }),
          activeProject: null,
          messages: [],
          chatId: "c",
          usage: undefined,
          refreshAll,
          setHistoryFile: vi.fn(),
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
        useChatPersistence({
          tab: makeTab(),
          activeProject: null,
          messages: [],
          chatId: "c",
          usage: undefined,
          refreshAll: async () => {},
          setHistoryFile: vi.fn(),
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

  describe("handleSaveSubmit", () => {
    it("saves a new chat and adopts its history file", async () => {
      const saveSpy = vi.fn(async () => undefined)
      const setHistoryFile = vi.fn()
      apiMockControls.__setImpl("saveChatHistory", saveSpy)
      const messages = [{ role: "user" as const, content: "q" }]

      const { result } = renderHook(() =>
        useChatPersistence({
          tab: makeTab(),
          activeProject: null,
          messages,
          chatId: "c",
          usage: undefined,
          refreshAll: async () => {},
          setHistoryFile,
        }),
      )

      act(() => result.current.setSaveModalOpen(true))
      await act(async () => { await result.current.handleSaveSubmit("notes") })

      expect(saveSpy).toHaveBeenCalledWith("notes.json", messages, { chatId: "c", title: "notes", usage: undefined })
      expect(setHistoryFile).toHaveBeenCalledWith("notes.json")
      expect(result.current.saveModalOpen).toBe(false)
    })
  })
})
