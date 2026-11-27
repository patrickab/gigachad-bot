import { afterEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, renderHook, waitFor } from "@testing-library/react"
import type { ReactNode } from "react"
import { DEFAULT_TAB_CONFIG } from "@/contexts/SettingsContext"
import { TabContext, type TabContextValue } from "@/contexts/TabContext"
import type { ChatInputHandle } from "@/features/chat/composer/ChatInput"
import type { ChatReveal } from "@/features/chat/ChatContainer"
import type { SSEEvent, SSEStreamResult } from "@/lib/backend/sse"
import type { ChatHistory } from "@/lib/domain/histories"
import type { Attachment, ChatRequest, Message } from "@/lib/domain/types"
import { useTabChat } from "../useTabChat"

/** A tab's chat deletes and edits pairs as `useTabChat` wires them: the list, the saved history file and the composer. */

const { saveChat, streams, history, branches, undoDelete, enabledTools } = vi.hoisted(() => ({
  saveChat: vi.fn<(...args: unknown[]) => Promise<void>>(async () => {}),
  streams: { requests: [] as unknown[], fail: false },
  history: { current: null as unknown },
  branches: {
    refreshAll: async () => {},
    createBranch: async () => ({ childFile: "child.json", chatId: "child" }),
    setActiveFile: () => {},
    setActiveQaIndex: () => {},
  },
  undoDelete: { schedule: () => {}, pending: () => false },
  enabledTools: { enabledTools: [], toggleTool: () => {} },
}))

/** Answers "edited answer" and completes, or fails before any event when `streams.fail`. */
function stubStream(req: unknown): SSEStreamResult {
  streams.requests.push(req)
  const events: SSEEvent[] = [{ event: "token", data: "edited answer" }, { event: "done", data: "" }]
  return {
    abort: () => {},
    async *[Symbol.asyncIterator]() {
      if (streams.fail) throw new Error("model unavailable")
      yield* events
    },
  }
}

vi.mock("@/lib/backend/api", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/lib/backend/api")>(),
  loadChatHistory: async () => history.current,
  createChatStream: (req: unknown) => stubStream(req),
  saveChat,
  writeDocument: async () => {},
}))
vi.mock("@/hooks/useResource", () => ({
  useResource: () => undefined,
  useResourceState: () => ({ value: undefined, error: null }),
  useResourceFamily: () => ({}),
}))
vi.mock("@/features/navigation/BranchContext", () => ({ useBranches: () => branches }))
vi.mock("@/contexts/UndoDeleteContext", () => ({ useUndoDelete: () => undoDelete }))
vi.mock("@/features/chat/EnabledToolsContext", () => ({ useEnabledTools: () => enabledTools }))
vi.mock("@/contexts/SettingsContext", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/contexts/SettingsContext")>(),
  useSettings: () => ({ ocrModel: "" }),
  useNeovimSetting: () => ({ available: false, enabled: false, setPreferred: () => {} }),
}))

const notes: Attachment = { name: "notes.md", mime: "text/markdown", url: "/notes.md", content: "notes", active: true }
const messages: Message[] = [
  { role: "user", content: "Question 0" },
  { role: "assistant", content: "Answer 0" },
  { role: "user", content: "Question 1", attachments: [notes] },
  { role: "assistant", content: "Answer 1" },
  { role: "user", content: "Question 2" },
  { role: "assistant", content: "Answer 2" },
]

const noop = () => {}
const reveal: ChatReveal = { historyFile: null, qaIndex: null, seq: 0 }

/** `useTabChat` for a chat holding `messages`, saved in `historyFile` or unsaved, with a stub composer. */
async function renderChat(historyFile: string | null) {
  history.current = {
    messages, filename: "c1.json", chat_id: "c1", title: null, usage: null, parent_id: null, branch_message_idx: null, children: [],
  } satisfies ChatHistory
  const tab: TabContextValue = {
    tab: {
      id: "t1", name: null, chatId: "c1", historyFile, title: null,
      config: { ...DEFAULT_TAB_CONFIG, selectedModel: "m" }, appMode: "chat", projectSlug: null, workspaceItem: null,
    },
    isActive: true,
    setModeLabel: noop, updateConfig: noop, setAppMode: noop, setWorkspaceItem: noop, setHistoryFile: noop, setTitle: noop,
  }
  const wrapper = ({ children }: { children: ReactNode }) => <TabContext.Provider value={tab}>{children}</TabContext.Provider>
  const rendered = renderHook(() => useTabChat({ reveal, onOpenChat: noop, onOpenItem: noop }), { wrapper })
  if (historyFile) await waitFor(() => expect(rendered.result.current.chat.messages).toHaveLength(messages.length))
  else act(() => { rendered.result.current.chat.replaceMessages(() => messages) })
  let draft = ""
  const composer = { updateText: (update: (current: string) => string) => { draft = update(draft) } } as ChatInputHandle
  rendered.result.current.chatInputRef.current = composer
  return { ...rendered, draft: () => draft }
}

const contents = (list: Message[]) => list.map((m) => m.content)
const lastSaved = () => saveChat.mock.calls.at(-1)?.[2] as Message[]

afterEach(() => {
  cleanup()
  saveChat.mockClear()
  streams.requests = []
  streams.fail = false
})

describe("deleting a pair", () => {
  it("removes it from the chat and saves the history", async () => {
    const { result } = await renderChat("c1.json")

    act(() => result.current.turns.handleDeletePair(2))

    await waitFor(() => expect(contents(result.current.chat.messages)).toEqual(["Question 0", "Answer 0", "Question 2", "Answer 2"]))
    expect(saveChat).toHaveBeenCalledTimes(1)
    expect(saveChat.mock.calls[0][0]).toBe("c1.json")
    expect(contents(lastSaved())).toEqual(["Question 0", "Answer 0", "Question 2", "Answer 2"])
  })

  it("keeps the pair and says why when the history file cannot be written", async () => {
    const { result } = await renderChat("c1.json")
    saveChat.mockRejectedValueOnce(new Error("disk full"))

    act(() => result.current.turns.handleDeletePair(2))

    await waitFor(() => expect(result.current.turns.turnError).toBe("The chat could not be saved, so the turn was not deleted."))
    expect(contents(result.current.chat.messages)).toEqual(contents(messages))

    // The next delete that saves clears it
    act(() => result.current.turns.handleDeletePair(2))
    await waitFor(() => expect(contents(result.current.chat.messages)).toEqual(["Question 0", "Answer 0", "Question 2", "Answer 2"]))
    expect(result.current.turns.turnError).toBeNull()
  })
})

describe("editing a pair", () => {
  it("removes the later pairs and sends the edit once, with the pair's attachments", async () => {
    const { result } = await renderChat("c1.json")

    act(() => result.current.onEditPair(2, "Question 1, edited"))

    await waitFor(() => expect(contents(lastSaved() ?? [])).toContain("edited answer"))
    await waitFor(() => expect(contents(result.current.chat.messages)).toEqual(["Question 0", "Answer 0", "Question 1, edited", "edited answer"]))
    expect(result.current.chat.messages[2].attachments?.map((a) => a.name)).toEqual(["notes.md"])
    expect(contents(saveChat.mock.calls[0][2] as Message[])).toEqual(["Question 0", "Answer 0"])
    expect(streams.requests).toHaveLength(1)
    const request = streams.requests[0] as ChatRequest & { messages: { content: string }[] }
    expect(request.user_msg).toContain("notes")
    expect(request.user_msg).toContain("Question 1, edited")
    expect(request.messages.map((m) => m.content).slice(0, 2)).toEqual(["Question 0", "Answer 0"])
    expect(request.messages).toHaveLength(3)
  })

  it("puts the edited text into the composer when the send fails", async () => {
    streams.fail = true
    const { result, draft } = await renderChat("c1.json")

    act(() => result.current.onEditPair(4, "Question 2, edited"))

    await waitFor(() => expect(draft()).toBe("Question 2, edited"))
    expect(contents(saveChat.mock.calls[0][2] as Message[])).toEqual(["Question 0", "Answer 0", "Question 1", "Answer 1"])
  })

  it("keeps every pair, sends nothing and says why when the cut cannot be saved", async () => {
    const { result, draft } = await renderChat("c1.json")
    saveChat.mockRejectedValueOnce(new Error("disk full"))

    act(() => result.current.onEditPair(2, "Question 1, edited"))

    await waitFor(() => expect(result.current.turns.turnError).toBe("The chat could not be saved, so the turn was not edited."))
    expect(contents(result.current.chat.messages)).toEqual(contents(messages))
    expect(streams.requests).toHaveLength(0)
    expect(draft()).toBe("")
  })
})
