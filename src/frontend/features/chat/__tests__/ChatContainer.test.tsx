import { afterEach, beforeAll, describe, expect, it, vi } from "vitest"
import { cleanup, render, screen } from "@testing-library/react"
import { ChatContainer } from "../ChatContainer"

// jsdom has no layout or scrolling, the chat pins itself to the bottom on mount
beforeAll(() => { Element.prototype.scrollIntoView = () => {} })
vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} })

vi.mock("@/features/chat/EnabledToolsContext", () => ({ useEnabledTools: () => ({ enabledTools: [], toggleTool: () => {} }) }))
vi.mock("@/contexts/SettingsContext", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/contexts/SettingsContext")>(),
  useSettings: () => ({ ocrModel: "" }),
  useNeovimSetting: () => ({ available: false, enabled: false, setPreferred: () => {} }),
}))
// jsdom lays nothing out: give the column and composer a size so the empty state can center
vi.mock("@/hooks/useElementSize", () => ({ useElementSize: () => ({ width: 800, height: 600 }) }))

const GREETING = "What's on your mind?"

function renderChat(historyLoading: boolean) {
  render(<ChatContainer chatId="c1" messages={[]} historyLoading={historyLoading} isStreaming={false} onSend={() => {}} onCancel={() => {}} onDeletePair={() => {}} />)
  return { greeting: screen.getByText(GREETING), composerArea: screen.getByText(GREETING).parentElement! }
}

describe("ChatContainer empty state", () => {
  afterEach(cleanup)

  it("centers the composer under the greeting in a chat known to be empty", () => {
    const { greeting, composerArea } = renderChat(false)
    expect(greeting).toHaveAttribute("aria-hidden", "false")
    expect(composerArea.style.transform).not.toBe("translateY(0px)")
  })

  it("keeps the composer at the bottom without a greeting while the history loads", () => {
    const { greeting, composerArea } = renderChat(true)
    expect(greeting).toHaveAttribute("aria-hidden", "true")
    expect(composerArea.style.transform).toBe("translateY(0px)")
  })
})
