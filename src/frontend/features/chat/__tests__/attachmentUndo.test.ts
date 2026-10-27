import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { DEFAULT_TAB_CONFIG } from "@/contexts/SettingsContext"
import { useChatStream, type UseChatStreamReturn } from "@/features/chat/useChatStream"
import { useChatTurns } from "@/features/chat/useChatTurns"
import type * as Api from "@/lib/backend/api"
import type { Attachment, Message } from "@/lib/domain/types"

const { schedule } = vi.hoisted(() => ({ schedule: vi.fn() }))

vi.mock("@/contexts/UndoDeleteContext", () => ({ useUndoDelete: () => ({ schedule, pending: () => false }) }))
vi.mock("@/features/chat/EnabledToolsContext", () => ({ useEnabledTools: () => ({ enabledTools: [] }) }))
vi.mock("@/lib/backend/api", async (importOriginal) => ({
  ...await importOriginal<typeof Api>(),
  createChatStream: vi.fn(),
  deleteAttachment: vi.fn(),
}))

function attachment(name: string, content: string): Attachment {
  return { name, mime: "text/markdown", url: "", content, active: true }
}

const notes = attachment("notes.md", "notes")
const plan = attachment("plan.md", "plan")

function renderChat() {
  return renderHook(() => {
    const chat = useChatStream()
    const turns = useChatTurns({
      chat,
      chatId: "chat-1",
      projectSlug: null,
      config: DEFAULT_TAB_CONFIG,
      reasoningEffort: "none",
      prompts: {},
      autosave: async () => {},
      refreshDocuments: () => {},
      runSlashCommand: async () => false,
    })
    return { chat, turns }
  })
}

/** A chat whose first user message held `notes` then `plan`, after removing `notes`; returns the scheduled undo. */
function chatWithRemovedNotes() {
  const rendered = renderChat()
  const user: Message = { role: "user", content: "read these", attachments: [notes, plan] }
  act(() => { rendered.result.current.chat.replaceMessages(() => [user, { role: "assistant", content: "done" }]) })
  act(() => rendered.result.current.turns.handleRemoveAttachment(0, "notes.md"))
  expect(rendered.result.current.chat.messages[0].attachments).toEqual([plan])
  const undo = schedule.mock.calls[0][3] as () => void
  return { ...rendered, undo }
}

/** Swap the first message for `replace(it)`, as a reference sync or edit does during the undo window. */
function replaceFirstMessage(chat: () => UseChatStreamReturn, replace: (message: Message) => Message): Message {
  const replacement = replace(chat().messages[0])
  act(() => { chat().replaceMessages((current) => current.map((m, i) => i === 0 ? replacement : m)) })
  return replacement
}

beforeEach(() => {
  schedule.mockReset()
})

describe("attachment removal undo", () => {
  it("re-inserts the attachment at its old position when the message was replaced meanwhile", () => {
    const { result, undo } = chatWithRemovedNotes()
    replaceFirstMessage(() => result.current.chat, (message) => ({ ...message, content: "read these, edited" }))

    act(() => undo())

    const restored = result.current.chat.messages[0]
    expect(restored.content).toBe("read these, edited")
    expect(restored.attachments).toEqual([notes, plan])
    expect(restored.hiddenContent).toContain("notes")
  })

  it("leaves the message alone when it already holds an attachment with that name", () => {
    const { result, undo } = chatWithRemovedNotes()
    const reattached = attachment("notes.md", "notes, attached again")
    const replacement = replaceFirstMessage(() => result.current.chat, (message) => ({ ...message, attachments: [plan, reattached] }))

    act(() => undo())

    expect(result.current.chat.messages[0]).toBe(replacement)
    expect(result.current.chat.messages[0].attachments).toEqual([plan, reattached])
  })
})
