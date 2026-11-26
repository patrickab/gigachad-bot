import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { DEFAULT_TAB_CONFIG } from "@/contexts/SettingsContext"
import type { Tab } from "@/contexts/TabContext"
import type { Message } from "@/lib/domain/types"
import { useChatPersistence } from "@/features/chat/useChatPersistence"
import { useChatStream } from "@/features/chat/useChatStream"
import { useChatTurns } from "@/features/chat/useChatTurns"

const api = vi.hoisted(() => ({ saveChat: vi.fn<(...args: unknown[]) => Promise<undefined>>(async () => undefined) }))

vi.mock("@/lib/backend/api", () => ({ saveChat: api.saveChat, deleteAttachment: vi.fn(), parseFiles: vi.fn(), writeDocument: vi.fn() }))
vi.mock("@/contexts/UndoDeleteContext", () => ({ useUndoDelete: () => ({ schedule: () => {}, pending: () => false }) }))
vi.mock("@/features/chat/EnabledToolsContext", () => ({ useEnabledTools: () => ({ enabledTools: [] }) }))

const tab = { id: "t1", name: "Tab", chatId: "c1", historyFile: "saved.json", title: "Saved" } as unknown as Tab
const initial: Message[] = [
  { role: "user", content: "q1" },
  { role: "assistant", content: "a1" },
  { role: "user", content: "q2" },
  { role: "assistant", content: "a2" },
]

beforeEach(() => api.saveChat.mockReset().mockResolvedValue(undefined))

describe("deleting a pair", () => {
  it("counts as deleted when the file was written and only the refresh failed, and an autosave does not bring it back", async () => {
    const { result } = renderHook(() => {
      const chat = useChatStream()
      const persistence = useChatPersistence({
        tab, projectSlug: null, messages: chat.messages, chatId: "c1", usage: undefined,
        refreshAll: async () => { throw new Error("refresh failed") }, setHistoryFile: vi.fn(),
      })
      const turns = useChatTurns({
        chat, chatId: "c1", projectSlug: null, config: DEFAULT_TAB_CONFIG, reasoningEffort: "none", prompts: {},
        save: persistence.handleSave, autosave: persistence.handleAutosave,
      })
      return { chat, persistence, turns }
    })

    act(() => result.current.chat.replaceMessages(() => initial))
    act(() => result.current.turns.handleDeletePair(0))
    await vi.waitFor(() => expect(result.current.chat.messages.map((m) => m.content)).toEqual(["q2", "a2"]))
    expect(result.current.turns.turnError).toBeNull()

    await act(async () => { await result.current.persistence.handleAutosave(result.current.chat.messages, { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }) })
    const written = api.saveChat.mock.calls.map((call) => (call[2] as Message[]).map((m) => m.content))
    expect(written).toEqual([["q2", "a2"], ["q2", "a2"]])
  })
})
