import { Profiler, type ComponentProps } from "react"
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { DEFAULT_TAB_CONFIG } from "@/contexts/SettingsContext"
import { TabContext, type TabContextValue } from "@/contexts/TabContext"
import type { ChatReveal } from "@/features/chat/ChatContainer"
import type { SSEEvent, SSEStreamResult } from "@/lib/backend/sse"
import type { ChatHistory } from "@/lib/domain/histories"
import type { Attachment, Message } from "@/lib/domain/types"
import { SidebarProvider } from "@/features/navigation/SidebarContext"
import { TabChat, TabContent } from "../TabContent"
import { useTabChat } from "../useTabChat"

/**
 * A streamed frame re-renders the live turn only, wired as the app wires it: `useTabChat`'s real outputs, passed to
 * `TabChat` the way `TabContent` passes them; `ChatContainer` renders the pairs on the shared `ConversationColumn`.
 * Only the network (`@/lib/backend/api`, the shared resources) and the app-wide providers (tab, branches, undo, tools,
 * settings) are stubbed, with values as stable as the real ones.
 *
 * Props covered, every one `TabContent` hands `TabChat` (add a new one here and to `ChatOfTab` below):
 * chatId, historyFile, draftKey, onChatColumnChange, chat (messages, isStreaming, cancel), historyLoading, onDeletePair, onEditPair, extracting,
 * onSend, onBranch, reveal, branchMessageIdx, chatInputRef, slug, attachmentActions
 * (onToggleActive, onRemove, onContentChange, onPreviewOpen), attachMenuItems, composerLeading, composerNotice.
 * What `ChatContainer` hands a pair from them: the pair's messages and indexes, isStreaming, onCollapse,
 * onAttachmentClick (its preview opener), onDelete, onEdit, laterTurnsOf, onBranch, the branch separator.
 * Edit and delete pause through `TurnEditsPaused` while a reply streams: no prop of a settled pair changes when a send starts or ends.
 * Text is counted where it renders (`LaTeXMarkdown`, by content), a question's attachments where they render (`MessageAttachments`, by names).
 */

const { renders, durations, stream, history, branches, undoDelete, enabledTools, project } = vi.hoisted(() => ({
  renders: new Map<string, number>(),
  /** Summed `actualDuration` per `<Profiler>` id, while `durations.measuring` */
  durations: { measuring: false, ms: new Map<string, number>() },
  stream: { current: null as null | { push: (event: SSEEvent) => void } },
  history: { current: null as unknown },
  branches: {
    refreshAll: async () => {},
    createBranch: async () => ({ childFile: "child.json", chatId: "child" }),
  },
  undoDelete: { schedule: () => {}, pending: () => false },
  enabledTools: { enabledTools: [], toggleTool: () => {} },
  project: { projects: [] },
}))

beforeAll(() => { Element.prototype.scrollIntoView = () => {} })
vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} })
// The composer keeps its draft there, jsdom under this Node has none
const stored = new Map<string, string>()
vi.stubGlobal("localStorage", {
  getItem: (key: string) => stored.get(key) ?? null,
  setItem: (key: string, value: string) => { stored.set(key, value) },
  removeItem: (key: string) => { stored.delete(key) },
})

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
vi.mock("@/features/navigation/BranchContext", () => ({ useBranchActions: () => branches }))
vi.mock("@/contexts/UndoDeleteContext", () => ({ useUndoDelete: () => undoDelete }))
vi.mock("@/features/chat/EnabledToolsContext", () => ({ useEnabledTools: () => enabledTools }))
vi.mock("@/contexts/ProjectContext", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/contexts/ProjectContext")>(),
  useProject: () => project,
}))
vi.mock("@/contexts/SettingsContext", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/contexts/SettingsContext")>(),
  useSettings: () => ({ ocrModel: "" }),
  useNeovimSetting: () => ({ available: false, enabled: false, setPreferred: () => {} }),
}))
const count = vi.hoisted(() => (key: string) => { renders.set(key, (renders.get(key) ?? 0) + 1) })
vi.mock("@/components/content/LaTeXMarkdown", () => ({
  LaTeXMarkdown: ({ content }: { content: string }) => {
    count(content)
    return <div>{content}</div>
  },
}))
const profile = vi.hoisted(() => (id: string, _phase: string, actualDuration: number) => {
  if (durations.measuring) durations.ms.set(id, (durations.ms.get(id) ?? 0) + actualDuration)
})
// The top bar, the composer, the surface and the tab workspace, counted and profiled around the real components
vi.mock("@/features/navigation/TopBar", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/features/navigation/TopBar")>()
  const { Profiler } = await import("react")
  return { ...actual, TopBar: (props: ComponentProps<typeof actual.TopBar>) => <Profiler id="bar" onRender={profile}><actual.TopBar {...props} /></Profiler> }
})
vi.mock("@/features/settings/ModelDropdown", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/features/settings/ModelDropdown")>()
  return { ...actual, ModelDropdown: (props: ComponentProps<typeof actual.ModelDropdown>) => { count("ModelDropdown"); return <actual.ModelDropdown {...props} /> } }
})
vi.mock("@/features/chat/composer/ChatInput", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/features/chat/composer/ChatInput")>()
  const { forwardRef, Profiler } = await import("react")
  const ChatInput = forwardRef<import("@/features/chat/composer/ChatInput").ChatInputHandle, ComponentProps<typeof actual.ChatInput>>((props, ref) => (
    <Profiler id="composer" onRender={profile}><actual.ChatInput {...props} ref={ref} /></Profiler>
  ))
  return { ...actual, ChatInput }
})
// Counted inside the composer: only a render of `ChatInput`'s own body renders its textarea
vi.mock("@/components/ui/Composer", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/components/ui/Composer")>()
  return { ...actual, ComposerTextarea: (props: ComponentProps<typeof actual.ComposerTextarea>) => { count("ChatInput"); return <actual.ComposerTextarea {...props} /> } }
})
vi.mock("../WorkspaceSurface", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../WorkspaceSurface")>()
  return { ...actual, WorkspaceSurface: (props: ComponentProps<typeof actual.WorkspaceSurface>) => { count("WorkspaceSurface"); return <actual.WorkspaceSurface {...props} /> } }
})
// `TabContent`'s workspace calls `useTabChat` once per render
vi.mock("../useTabChat", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../useTabChat")>()
  return { ...actual, useTabChat: (...args: Parameters<typeof actual.useTabChat>) => { count("TabWorkspace"); return actual.useTabChat(...args) } }
})
vi.mock("@/features/chat/messages/MessageAttachments", () => ({
  MessageAttachments: ({ attachments }: { attachments: Attachment[] }) => {
    count(`attachments:${attachments.map((attachment) => attachment.name).join(",")}`)
    return null
  },
}))

const notes: Attachment = { name: "notes.md", mime: "text/markdown", url: "/notes.md", content: "notes", active: true }
const settled: Message[] = [
  { role: "user", content: "Question 0", attachments: [notes] },
  { role: "assistant", content: "Answer 0" },
  { role: "user", content: "Question 1" },
  { role: "assistant", content: "Answer 1" },
  { role: "user", content: "Question 2" },
  { role: "assistant", content: "Answer 2" },
]
const shortHistory = {
  messages: settled, filename: "c1.json", chat_id: "c1", title: null, usage: null, parent_id: null, branch_message_idx: 0, children: [],
} satisfies ChatHistory
history.current = shortHistory

const noop = () => {}
const tab: TabContextValue = {
  tab: {
    id: "t1", chatId: "c1", historyFile: "c1.json", title: null,
    config: { ...DEFAULT_TAB_CONFIG, selectedModel: "m" }, projectSlug: null, workspaceItem: null, itemHidden: false,
  },
  isActive: true,
  setModeLabel: noop, updateConfig: noop, hideItem: noop, dropItem: noop, setWorkspaceItem: noop, setHistoryFile: noop, setTitle: noop,
}
const reveal: ChatReveal = { historyFile: null, qaIndex: null, seq: 0 }
const onOpenChat = () => {}
const onOpenItem = () => {}

/** `TabContent`'s chat body for a non-empty chat: `useTabChat`'s outputs handed down prop for prop. */
function ChatOfTab({ onTabChat }: { onTabChat: (tabChat: ReturnType<typeof useTabChat>) => void }) {
  const tabChat = useTabChat({ onOpenChat, onOpenItem })
  onTabChat(tabChat)
  const { chat, chatId, branchMessageIdx, historyLoading, extracting, chatInputRef, docs, turns, attachmentActions, attachMenuItems, onEditPair, onBranch } = tabChat
  return (
    <TabChat
      chat={chat}
      chatId={chatId}
      historyFile={tab.tab.historyFile ?? null}
      draftKey={`chat::${tab.tab.historyFile}`}
      onChatColumnChange={noop}
      historyLoading={historyLoading}
      onDeletePair={turns.handleDeletePair}
      onEditPair={onEditPair}
      extracting={extracting}
      onSend={(text, attachments) => void turns.handleSend(text, attachments)}
      onBranch={onBranch}
      reveal={reveal}
      branchMessageIdx={branchMessageIdx}
      chatInputRef={chatInputRef}
      slug={null}
      attachmentActions={attachmentActions}
      attachMenuItems={attachMenuItems}
      composerLeading={undefined}
      composerNotice={turns.turnError || docs.architectureError || undefined}
    />
  )
}

describe("a tab's chat while a reply streams", () => {
  afterEach(() => {
    cleanup()
    stream.current = null
    renders.clear()
    history.current = shortHistory
  })

  it("re-renders neither an open nor a collapsed settled pair", async () => {
    let tabChat!: ReturnType<typeof useTabChat>
    render(<TabContext.Provider value={tab}><ChatOfTab onTabChat={(current) => { tabChat = current }} /></TabContext.Provider>)
    fireEvent.click(await screen.findByRole("button", { name: "Question 0" }))
    expect(screen.getByRole("separator")).toBeTruthy()

    act(() => { void tabChat.turns.handleSend("Question 3", []) })
    await waitFor(() => expect(stream.current).not.toBeNull())
    act(() => stream.current!.push({ event: "token", data: "A" }))
    await waitFor(() => expect(renders.get("A")).toBeGreaterThan(0))
    expect(tabChat.chat.get().isStreaming).toBe(true)

    // Question 0 open with its attachment, Question 1 collapsed to its answer's head
    const watched = ["Question 0", "attachments:notes.md", "Answer 0", "Answer 1"]
    const before = watched.map((key) => renders.get(key) ?? 0)
    expect(before.every((count) => count > 0)).toBe(true)

    let answer = "A"
    for (const token of ["n", "s", "w", "e", "r"]) {
      answer += token
      act(() => stream.current!.push({ event: "token", data: token }))
      await waitFor(() => expect(renders.get(answer)).toBeGreaterThan(0))
    }
    expect(watched.map((key) => renders.get(key) ?? 0)).toEqual(before)

    act(() => stream.current!.push({ event: "done", data: "" }))
    await waitFor(() => expect(tabChat.chat.get().isStreaming).toBe(false))
  })

  it("re-renders no settled pair for a send from start to end, nor for the delete of a later pair", async () => {
    let tabChat!: ReturnType<typeof useTabChat>
    render(<TabContext.Provider value={tab}><ChatOfTab onTabChat={(current) => { tabChat = current }} /></TabContext.Provider>)
    fireEvent.click(await screen.findByRole("button", { name: "Question 0" }))
    const watched = ["Question 0", "attachments:notes.md", "Answer 0", "Answer 1"]
    const before = watched.map((key) => renders.get(key) ?? 0)
    expect(before.every((count) => count > 0)).toBe(true)

    act(() => { void tabChat.turns.handleSend("Question 3", []) })
    await waitFor(() => expect(stream.current).not.toBeNull())
    act(() => stream.current!.push({ event: "token", data: "Reply" }))
    act(() => stream.current!.push({ event: "done", data: "" }))
    await waitFor(() => expect(tabChat.chat.get().isStreaming).toBe(false))
    expect(watched.map((key) => renders.get(key) ?? 0)).toEqual(before)

    act(() => tabChat.turns.handleDeletePair(4))
    await waitFor(() => expect(tabChat.chat.get().messages.map((m) => m.content)).not.toContain("Question 2"))
    expect(watched.map((key) => renders.get(key) ?? 0)).toEqual(before)
  })

  it("renders the live pair's question per token frame (measure)", async () => {
    let tabChat!: ReturnType<typeof useTabChat>
    render(<TabContext.Provider value={tab}><ChatOfTab onTabChat={(current) => { tabChat = current }} /></TabContext.Provider>)
    await screen.findByRole("button", { name: "Question 0" })

    // None: the question keeps its props (it binds its index to the stable attachment handler). Was 20 before task pi-live.
    expect(await questionRendersPerTokenFrames(() => tabChat, 20)).toBe(0)
  })

  it("renders the top bar, the composer and the tab workspace per token frame (measure)", async () => {
    // 20 settled turns, in the tab's real workspace: `TabContent` with its surface, top bar and composer
    const messages = Array.from({ length: 20 }, (_, index): Message[] => [
      { role: "user", content: `Question ${index}` },
      { role: "assistant", content: `Answer ${index}` },
    ]).flat()
    history.current = { ...shortHistory, messages }
    render(
      <Profiler id="tab" onRender={profile}>
        <SidebarProvider>
          <TabContext.Provider value={tab}>
            <TabContent onOpenChat={noop} onOpenItem={onOpenItem} reveal={reveal} />
          </TabContext.Provider>
        </SidebarProvider>
      </Profiler>,
    )
    await waitFor(() => expect(renders.get("Answer 19")).toBeGreaterThan(0))
    const input = screen.getByPlaceholderText("Message")
    fireEvent.change(input, { target: { value: "Question 20" } })
    fireEvent.keyDown(input, { key: "Enter" })
    await waitFor(() => expect(stream.current).not.toBeNull())

    let answer = ""
    const frame = async () => {
      answer += "b"
      act(() => stream.current!.push({ event: "token", data: "b" }))
      await waitFor(() => expect(renders.get(answer)).toBeGreaterThan(0))
    }
    // The first two frames also settle the send
    await frame()
    await frame()
    const watched = ["ModelDropdown", "ChatInput", "WorkspaceSurface", "TabWorkspace"] as const
    const before = Object.fromEntries(watched.map((key) => [key, renders.get(key) ?? 0]))
    durations.ms.clear()
    durations.measuring = true
    for (let index = 0; index < 100; index++) await frame()
    durations.measuring = false
    const counts = Object.fromEntries(watched.map((key) => [key, (renders.get(key) ?? 0) - before[key]]))
    const { bar = 0, composer = 0, tab: total = 0 } = Object.fromEntries(durations.ms)
    console.info(`chat, 100 token frames: ${JSON.stringify(counts)}; bar ${(100 * bar / total).toFixed(1)}%, composer ${(100 * composer / total).toFixed(1)}% of ${total.toFixed(0)} ms`)

    // Only the chat body follows the messages: the workspace, its surface and the bar were 100 each before task LA3, the composer before task LA4
    expect(counts).toEqual({ ModelDropdown: 0, ChatInput: 0, WorkspaceSurface: 0, TabWorkspace: 0 })

    await act(async () => stream.current!.push({ event: "done", data: "" }))
  })
})

/**
 * Renders of the live pair's question (its `MessageAttachments`) over `frames` token frames, one token rendered per frame:
 * a send of "Question 3" with the attachment "live.md", then one-letter tokens.
 */
async function questionRendersPerTokenFrames(tabChat: () => ReturnType<typeof useTabChat>, frames: number): Promise<number> {
  const attachment: Attachment = { name: "live.md", mime: "text/markdown", url: "/live.md", content: "live", active: true }
  act(() => { void tabChat().turns.handleSend("Question 3", [attachment]) })
  await waitFor(() => expect(stream.current).not.toBeNull())
  let answer = ""
  const frame = async () => {
    answer += "b"
    act(() => stream.current!.push({ event: "token", data: "b" }))
    await waitFor(() => expect(renders.get(answer)).toBeGreaterThan(0))
  }
  // The first two frames also settle the send
  await frame()
  await frame()
  const key = "attachments:live.md"
  const before = renders.get(key) ?? 0
  expect(before).toBeGreaterThan(0)
  for (let index = 0; index < frames; index++) await frame()
  const count = (renders.get(key) ?? 0) - before
  act(() => stream.current!.push({ event: "done", data: "" }))
  await waitFor(() => expect(tabChat().chat.get().isStreaming).toBe(false))
  return count
}
