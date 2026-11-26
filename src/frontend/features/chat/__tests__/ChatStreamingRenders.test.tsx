import { afterEach, beforeAll, describe, expect, it, vi } from "vitest"
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react"
import { DEFAULT_TAB_CONFIG } from "@/contexts/SettingsContext"
import type { Message } from "@/lib/domain/types"
import { ChatContainer } from "../ChatContainer"
import { useChatStream, type UseChatStreamReturn } from "../useChatStream"
import { useChatTurns } from "../useChatTurns"

const { renders, enabledTools, undoDelete } = vi.hoisted(() => ({
  renders: new Map<string, number>(),
  enabledTools: { enabledTools: [], toggleTool: () => {} },
  undoDelete: { schedule: () => {}, pending: () => false },
}))

beforeAll(() => { Element.prototype.scrollIntoView = () => {} })
vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} })

vi.mock("@/contexts/UndoDeleteContext", () => ({ useUndoDelete: () => undoDelete }))
vi.mock("@/features/chat/EnabledToolsContext", () => ({ useEnabledTools: () => enabledTools }))
vi.mock("@/contexts/SettingsContext", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/contexts/SettingsContext")>(),
  useSettings: () => ({ ocrModel: "" }),
  useNeovimSetting: () => ({ available: false, enabled: false, setPreferred: () => {} }),
}))
// Counts the renders of each message by its text
vi.mock("@/features/chat/messages/ChatMessage", () => ({
  AssistantMessageContent: ({ content }: { content: string }) => <div>{content}</div>,
  ChatMessage: ({ content }: { content: string }) => {
    renders.set(content, (renders.get(content) ?? 0) + 1)
    return <div>{content}</div>
  },
}))

// Stable as a tab's are, so only the messages change per frame
const autosave = async () => {}
const prompts = {}

/** The chat as a tab wires it: the column gets the handlers of `useChatTurns`. */
function StreamingChat({ onChat }: { onChat: (chat: UseChatStreamReturn) => void }) {
  const chat = useChatStream()
  onChat(chat)
  const turns = useChatTurns({ chat, chatId: "c1", projectSlug: null, config: DEFAULT_TAB_CONFIG, reasoningEffort: "none", prompts, autosave })
  return (
    <ChatContainer
      chatId="c1"
      messages={chat.messages}
      isStreaming={chat.messages.length > 4}
      onSend={() => {}}
      onCancel={() => {}}
      onDeletePair={turns.handleDeletePair}
      onRegenerate={turns.handleRegenerate}
    />
  )
}

describe("ChatContainer while a reply streams", () => {
  afterEach(cleanup)

  it("does not re-render an open settled pair when the live answer grows", () => {
    let chat!: UseChatStreamReturn
    render(<StreamingChat onChat={(current) => { chat = current }} />)
    const settled: Message[] = [
      { role: "user", content: "Question 0" },
      { role: "assistant", content: "Answer 0" },
      { role: "user", content: "Question 1" },
      { role: "assistant", content: "Answer 1" },
    ]
    act(() => { chat.replaceMessages(() => settled) })
    fireEvent.click(screen.getByRole("button", { name: "Question 0" }))
    act(() => { chat.replaceMessages((current) => [...current, { role: "user", content: "Question 2" }, { role: "assistant", content: "A" }]) })
    const before = renders.get("Answer 0")
    expect(before).toBeGreaterThan(0)

    for (const tail of ["An", "Ans", "Answ"]) {
      act(() => { chat.replaceMessages((current) => [...current.slice(0, -1), { role: "assistant", content: tail }]) })
    }
    expect(screen.getByText("Answ")).toBeTruthy()
    expect(renders.get("Answer 0")).toBe(before)
  })
})
