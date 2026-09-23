import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { ReactElement } from "react"
import { ChatInput } from "@/components/ChatInput"
import { ModeProvider } from "@/hooks/useModeState"
import { SettingsProvider } from "@/contexts/SettingsContext"

const api = vi.hoisted(() => ({
  uploadFile: vi.fn(),
  saveNotebook: vi.fn(),
  fetchBackendConfig: vi.fn(),
  ApiError: class ApiError extends Error {
    status: number
    constructor(message: string, status: number) {
      super(message)
      this.name = "ApiError"
      this.status = status
    }
  },
}))

vi.mock("@/lib/api", () => api)

/** ChatInput needs only the composer chrome; the tool menu is the LayoutGrid button, the
 *  only icon-only control left of the mic button. */
function renderComposer(chatId: string) {
  render(
    <SettingsProvider>
      <ModeProvider>
        <ChatInput chatId={chatId} onSend={vi.fn()} />
      </ModeProvider>
    </SettingsProvider>,
  )
}

function openToolMenu() {
  // Menu state survives rerenders of the same chat input, so toggle only when closed.
  if (screen.queryByText("Jupyter Mode")) return
  const menuButton = screen.getAllByRole("button").find((b) => b.querySelector(".lucide-layout-grid"))
  if (!menuButton) throw new Error("tool menu button not found")
  fireEvent.click(menuButton)
}

function notebookRow() {
  return screen.getByText("Jupyter Mode")
}

async function activate(row: HTMLElement, expectedCalls = 1) {
  fireEvent.click(row)
  await waitFor(() => expect(api.saveNotebook).toHaveBeenCalledTimes(expectedCalls))
}

describe("ChatInput Jupyter mode activation", () => {
  beforeEach(() => {
    api.uploadFile.mockReset().mockResolvedValue({ name: "u", mime: "image/jpeg", url: "u", active: true })
    api.saveNotebook.mockReset().mockResolvedValue({ revision_id: "r1" })
    api.fetchBackendConfig.mockReset().mockRejectedValue(new Error("offline"))
  })

  it("sends exactly one starter PUT on toggle, even with a second click mid-flight", async () => {
    const { promise } = Promise.withResolvers<{ revision_id: string }>()
    api.saveNotebook.mockReturnValueOnce(promise) // hangs: any duplicate PUT would show
    renderComposer("chat-1")
    openToolMenu()
    const row = notebookRow()

    fireEvent.click(row)
    fireEvent.click(row)
    await activate(row)

    expect(api.saveNotebook).toHaveBeenCalledTimes(1)
    expect(api.saveNotebook).toHaveBeenCalledWith("chat-1", "# %% [markdown]\n# Notebook\n", {}, "")
  })

  it("does not re-PUT once active for the chat", async () => {
    renderComposer("chat-1")
    openToolMenu()
    await activate(notebookRow())

    fireEvent.click(notebookRow())
    fireEvent.click(notebookRow())
    await Promise.resolve()

    expect(api.saveNotebook).toHaveBeenCalledTimes(1)
  })

  it("treats 409 (notebook already exists) as activated", async () => {
    api.saveNotebook.mockRejectedValueOnce(new api.ApiError("stale revision", 409))
    renderComposer("chat-1")
    openToolMenu()
    await activate(notebookRow())

    fireEvent.click(notebookRow())
    await Promise.resolve()

    expect(api.saveNotebook).toHaveBeenCalledTimes(1)
  })

  it("resets per chat: switching chats starts inactive and PUTs once each", async () => {
    const view = (chatId: string): ReactElement => (
      <SettingsProvider>
        <ModeProvider>
          <ChatInput chatId={chatId} onSend={vi.fn()} />
        </ModeProvider>
      </SettingsProvider>
    )
    const { rerender } = render(view("chat-1"))
    openToolMenu()
    await activate(notebookRow())

    rerender(view("chat-2"))
    openToolMenu()
    await activate(notebookRow(), 2)

    expect(api.saveNotebook).toHaveBeenNthCalledWith(1, "chat-1", "# %% [markdown]\n# Notebook\n", {}, "")
    expect(api.saveNotebook).toHaveBeenNthCalledWith(2, "chat-2", "# %% [markdown]\n# Notebook\n", {}, "")
  })
})