import { act, cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, expect, it, vi } from "vitest"
import { LEARN_EMBED_CHANNEL, LEARN_PROTOCOL_VERSION, type LearnFrameMessage } from "@gigachad/chat-ui/learn-protocol"
import { TabContext, type Tab } from "@/contexts/TabContext"
import type { LearnView } from "@/lib/domain/types"
import type * as Settings from "@/contexts/SettingsContext"
import { LearnProvider } from "../LearnProvider"
import { LearnSurface } from "../LearnSurface"

const FRAME_URL = "http://127.0.0.1:30181"
vi.mock("@/contexts/SettingsContext", async (importOriginal) => ({
  ...await importOriginal<typeof Settings>(),
  useSettings: () => ({ hostAccess: true, learnharnessUrl: FRAME_URL }),
}))
vi.mock("@/hooks/useThemeName", () => ({ useThemeName: () => "light" }))
vi.mock("../useLearnSessions", () => ({ useLearnSessions: () => ({ sessions: [], error: null, refresh: vi.fn(), claim: vi.fn() }) }))
vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} })

/** A learn tab over a fake LearnHarness frame at `version`: what the shell posted to it, and a way to post from it. */
function mountLearnTab(view: LearnView, version = LEARN_PROTOCOL_VERSION) {
  const tab = { tab: { chatId: "c1", projectSlug: null } as Tab, isActive: true, setModeLabel: vi.fn(), updateConfig: vi.fn(), setAppMode: vi.fn(), setWorkspaceItem: vi.fn(), setHistoryFile: vi.fn(), setTitle: vi.fn() }
  const rendered = render(
    <LearnProvider onOpenItem={vi.fn()}>
      <TabContext.Provider value={tab}><LearnSurface view={view} onViewChange={vi.fn()} /></TabContext.Provider>
    </LearnProvider>,
  )
  const frame = rendered.container.querySelector("iframe")!
  const posted = vi.spyOn(frame.contentWindow!, "postMessage").mockImplementation(() => {})
  const fromFrame = (data: LearnFrameMessage | { type: "ready"; version: number }) => act(() => {
    window.dispatchEvent(new MessageEvent("message", { data: { channel: LEARN_EMBED_CHANNEL, ...data }, origin: FRAME_URL, source: frame.contentWindow }))
  })
  fromFrame({ type: "ready", version })
  /** Shell messages of `type` posted so far, without the channel. */
  const sent = (type: string) => posted.mock.calls.map(([envelope]) => envelope as { type: string; channel?: string })
    .filter((message) => message.type === type)
    .map((message) => Object.fromEntries(Object.entries(message).filter(([key]) => key !== "channel")))
  return { fromFrame, sent }
}

const scrim = () => document.querySelector("[data-learn-modal-scrim]")

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

it("lets the frame's microphone through for LearnHarness's own dictation", () => {
  mountLearnTab({ kind: "session", sessionId: "s1" })
  expect(document.querySelector("iframe")?.getAttribute("allow")).toContain("microphone")
})

it("dims the shell around a dialog in the frame, a click or Escape on it closes the dialog", () => {
  const { fromFrame, sent } = mountLearnTab({ kind: "session", sessionId: "s1" })
  expect(scrim()).toBeNull()

  fromFrame({ type: "modal", open: true })
  fireEvent.click(scrim()!)
  fireEvent.keyDown(document, { key: "Escape" })
  expect(sent("close-modal")).toEqual([{ type: "close-modal" }, { type: "close-modal" }])

  fromFrame({ type: "modal", open: false })
  expect(scrim()).toBeNull()
})

it("asks for a LearnHarness restart at another protocol version, without dimming for its dialogs", () => {
  const { fromFrame } = mountLearnTab({ kind: "new" }, 1)
  fromFrame({ type: "modal", open: true })

  expect(screen.getByRole("status")).toHaveTextContent("LearnHarness needs a restart")
  expect(scrim()).toBeNull()
})
