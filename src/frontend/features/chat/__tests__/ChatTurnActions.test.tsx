import { afterEach, beforeAll, describe, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react"
import type { Message } from "@/lib/domain/types"
import { ChatContainer } from "../ChatContainer"

/** A chat's question actions while a reply streams, the same as in Pi: copy only, on every pair. */

beforeAll(() => { Element.prototype.scrollIntoView = () => {} })
vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} })

vi.mock("@/components/content/LaTeXMarkdown", () => ({ LaTeXMarkdown: ({ content }: { content: string }) => <div>{content}</div> }))
vi.mock("@/features/chat/EnabledToolsContext", () => ({ useEnabledTools: () => ({ enabledTools: [], toggleTool: () => {} }) }))
vi.mock("@/contexts/SettingsContext", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/contexts/SettingsContext")>(),
  useSettings: () => ({ ocrModel: "" }),
  useNeovimSetting: () => ({ available: false, enabled: false, setPreferred: () => {} }),
}))

const messages: Message[] = [
  { role: "user", content: "Question 0" },
  { role: "assistant", content: "Answer 0" },
  { role: "user", content: "Question 1" },
  { role: "assistant", content: "Answer 1" },
]

function chat(isStreaming: boolean) {
  return <ChatContainer chatId="c1" messages={messages} isStreaming={isStreaming} onSend={() => {}} onCancel={() => {}} onDeletePair={() => {}} onEditPair={() => {}} />
}

/** The question actions of every open pair. */
const questionActions = () => screen.getAllByRole("group", { name: "Question actions" }).map((group) => within(group).getAllByRole("button").map((button) => button.getAttribute("aria-label")))

describe("a chat's question actions", () => {
  afterEach(cleanup)

  it("offers no edit and no delete on any pair while a reply streams, both again once it ends", () => {
    const { rerender } = render(chat(false))
    fireEvent.click(screen.getByRole("button", { name: "Question 0" }))
    const settled = ["Copy message", "Edit message", "Delete turn"]
    expect(questionActions()).toEqual([settled, settled])

    rerender(chat(true))
    // The settled pair keeps copy, the live one has none
    expect(questionActions()).toEqual([["Copy message"]])

    rerender(chat(false))
    expect(questionActions()).toEqual([settled, settled])
  })
})
