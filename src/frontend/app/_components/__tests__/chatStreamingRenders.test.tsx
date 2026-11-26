import { afterEach, beforeAll, describe, expect, it, vi } from "vitest"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { DEFAULT_TAB_CONFIG } from "@/contexts/SettingsContext"
import { TabContext, type TabContextValue } from "@/contexts/TabContext"
import { ChatContainer, type ChatReveal } from "@/features/chat/ChatContainer"
import type { SSEEvent, SSEStreamResult } from "@/lib/backend/sse"
import type { ChatHistory } from "@/lib/domain/histories"
import type { Attachment, Message } from "@/lib/domain/types"
import { useTabChat } from "../useTabChat"

/**
 * A streamed frame re-renders the live turn only, wired as the app wires it: `useTabChat`'s real outputs, passed to
 * `ChatContainer` the way `TabContent` passes them. Only the network (`@/lib/backend/api`, the shared resources) and the
 * app-wide providers (tab, branches, undo, tools, settings) are stubbed, with values as stable as the real ones.
 *
 * Props covered, every one `TabContent` hands `ChatContainer` (add a new one here and to `TabChat` below):
 * chatId, historyFile, draftKey, onChatColumnChange, messages, historyLoading, isStreaming, onDeletePair, extracting,
 * onSend, onCancel, onRegenerate, onBranch, reveal, branchMessageIdx, chatInputRef, slug, attachmentActions
 * (onToggleActive, onRemove, onContentChange, onPreviewOpen), attachMenuItems, composerLeading, composerNotice.
 * What `ChatContainer` hands a pair from them: the pair's messages and indexes, isStreaming, onCollapse,
 * onAttachmentClick (from attachmentActions.onPreviewOpen), onDelete, onRegenerate, onBranch, the branch separator.
 */

const { renders, attachmentClicks, stream, history, branches, undoDelete, enabledTools } = vi.hoisted(() => ({
  renders: new Map<string, number>(),
  attachmentClicks: new Map<string, unknown>(),
  stream: { current: null as null | { push: (event: SSEEvent) => void } },
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

beforeAll(() => { Element.prototype.scrollIntoView = () => {} })
vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} })

/** A chat stream the test feeds event by event. */
function controlledStream(): SSEStreamResult {
  const queue: SSEEvent[] = []
  let wake: (() => void) | null = null
  let closed = false
  stream.current = { push: (event) => { queue.push(event); wake?.() } }
  return {
    abort: () => { closed = true; wake?.() },
    async *[Symbol.asyncIterator]() {
      while (true) {
        const event = queue.shift()
        if (event) {
          yield event
          if (event.event === "done") return
          continue
        }
        if (closed) return
        await new Promise<void>((resolve) => { wake = resolve })
        wake = null
      }
    },
  }
}

vi.mock("@/lib/backend/api", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/lib/backend/api")>(),
  loadChatHistory: async () => history.current,
  createChatStream: () => controlledStream(),
  saveChat: async () => {},
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
// Counts the renders of each message by role and text: an open pair's messages, and a collapsed pair's answer head
vi.mock("@/features/chat/messages/ChatMessage", () => ({
  AssistantMessageContent: ({ content }: { content: string }) => {
    renders.set(`preview:${content}`, (renders.get(`preview:${content}`) ?? 0) + 1)
    return <div>{content}</div>
  },
  ChatMessage: ({ role, content, onAttachmentClick }: { role: string; content: string; onAttachmentClick?: unknown }) => {
    renders.set(`${role}:${content}`, (renders.get(`${role}:${content}`) ?? 0) + 1)
    attachmentClicks.set(`${role}:${content}`, onAttachmentClick)
    return <div>{content}</div>
  },
}))

const notes: Attachment = { name: "notes.md", mime: "text/markdown", url: "/notes.md", content: "notes", active: true }
const settled: Message[] = [
  { role: "user", content: "Question 0", attachments: [notes] },
  { role: "assistant", content: "Answer 0" },
  { role: "user", content: "Question 1" },
  { role: "assistant", content: "Answer 1" },
]
history.current = {
  messages: settled, filename: "c1.json", chat_id: "c1", title: null, usage: null, parent_id: null, branch_message_idx: 0, children: [],
} satisfies ChatHistory

const noop = () => {}
const tab: TabContextValue = {
  tab: {
    id: "t1", name: null, chatId: "c1", historyFile: "c1.json", title: null,
    config: { ...DEFAULT_TAB_CONFIG, selectedModel: "m" }, appMode: "chat", projectSlug: null, workspaceItem: null,
  },
  isActive: true,
  setModeLabel: noop, updateConfig: noop, setAppMode: noop, setWorkspaceItem: noop, setHistoryFile: noop, setTitle: noop,
}
const reveal: ChatReveal = { historyFile: null, qaIndex: null, seq: 0 }
const onOpenChat = () => {}
const onOpenItem = () => {}

/** `TabContent`'s chat body for a non-empty chat: `useTabChat`'s outputs handed down prop for prop. */
function TabChat({ onTabChat }: { onTabChat: (tabChat: ReturnType<typeof useTabChat>) => void }) {
  const tabChat = useTabChat({ reveal, onOpenChat, onOpenItem })
  onTabChat(tabChat)
  const { chat, chatId, branchMessageIdx, historyLoading, extracting, chatInputRef, docs, turns, attachmentActions, attachMenuItems, onBranch } = tabChat
  return (
    <ChatContainer
      chatId={chatId}
      historyFile={tab.tab.historyFile}
      draftKey={`chat::${tab.tab.historyFile}`}
      onChatColumnChange={noop}
      messages={chat.messages}
      historyLoading={historyLoading}
      isStreaming={chat.isStreaming}
      onDeletePair={turns.handleDeletePair}
      extracting={extracting}
      onSend={(text, attachments) => turns.handleSend(text, attachments)}
      onCancel={chat.cancel}
      onRegenerate={turns.handleRegenerate}
      onBranch={onBranch}
      reveal={reveal}
      branchMessageIdx={branchMessageIdx}
      chatInputRef={chatInputRef}
      slug={null}
      attachmentActions={attachmentActions}
      attachMenuItems={attachMenuItems}
      composerLeading={undefined}
      composerNotice={docs.architectureError || undefined}
    />
  )
}

describe("a tab's chat while a reply streams", () => {
  afterEach(cleanup)

  it("re-renders neither an open nor a collapsed settled pair", async () => {
    let tabChat!: ReturnType<typeof useTabChat>
    render(<TabContext.Provider value={tab}><TabChat onTabChat={(current) => { tabChat = current }} /></TabContext.Provider>)
    fireEvent.click(await screen.findByRole("button", { name: "Question 0" }))
    expect(screen.getByRole("separator")).toBeTruthy()

    act(() => { void tabChat.turns.handleSend("Question 2", []) })
    await waitFor(() => expect(stream.current).not.toBeNull())
    act(() => stream.current!.push({ event: "token", data: "A" }))
    await waitFor(() => expect(renders.get("assistant:A")).toBeGreaterThan(0))
    expect(tabChat.chat.isStreaming).toBe(true)

    // Question 0 open with its attachment preview wired, Question 1 collapsed to its answer's head
    expect(attachmentClicks.get("user:Question 0")).toBeTypeOf("function")
    const watched = ["user:Question 0", "assistant:Answer 0", "preview:Answer 1"]
    const before = watched.map((key) => renders.get(key) ?? 0)
    expect(before.every((count) => count > 0)).toBe(true)

    let answer = "A"
    for (const token of ["n", "s", "w", "e", "r"]) {
      answer += token
      act(() => stream.current!.push({ event: "token", data: token }))
      await waitFor(() => expect(renders.get(`assistant:${answer}`)).toBeGreaterThan(0))
    }
    expect(watched.map((key) => renders.get(key) ?? 0)).toEqual(before)

    act(() => stream.current!.push({ event: "done", data: "" }))
    await waitFor(() => expect(tabChat.chat.isStreaming).toBe(false))
  })
})
