import { act, cleanup, fireEvent, render, screen } from "@testing-library/react"
import { useEffect } from "react"
import { afterEach, expect, it, vi } from "vitest"
import { LEARN_EMBED_CHANNEL, LEARN_PROTOCOL_VERSION, type LearnFrameMessage } from "@gigachad/chat-ui/learn-protocol"
import { TabContext, type Tab } from "@/contexts/TabContext"
import type { LearnView } from "@/lib/domain/types"
import type * as Settings from "@/contexts/SettingsContext"
import { LearnProvider, useLearn } from "../LearnProvider"
import { LearnPlanButton } from "../LearnPlanButton"

const FRAME_URL = "http://127.0.0.1:30181"
vi.mock("@/contexts/SettingsContext", async (importOriginal) => ({
  ...await importOriginal<typeof Settings>(),
  useSettings: () => ({ hostAccess: true, learnharnessUrl: FRAME_URL }),
}))
vi.mock("@/hooks/useThemeName", () => ({ useThemeName: () => "light" }))
vi.mock("../useLearnSessions", () => ({ useLearnSessions: () => ({ sessions: [], error: null, refresh: vi.fn(), claim: vi.fn() }) }))
vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} })

function Attach({ view }: { view: LearnView }) {
  const { attachFrame } = useLearn()
  useEffect(() => attachFrame({ slot: document.body, view, onViewChange: vi.fn() }), [attachFrame, view])
  return null
}

/** The learn top bar's Plan button on a tab showing `view`, over a fake frame. */
function mountPlanButton(view: LearnView) {
  const tab = { tab: { chatId: "c1", projectSlug: "p1" } as Tab, isActive: true, setModeLabel: vi.fn(), updateConfig: vi.fn(), setAppMode: vi.fn(), setWorkspaceItem: vi.fn(), setHistoryFile: vi.fn(), setTitle: vi.fn() }
  const onOpenItem = vi.fn()
  const rendered = render(
    <LearnProvider onOpenItem={onOpenItem}>
      <Attach view={view} />
      <TabContext.Provider value={tab}>
        <LearnPlanButton view={view} />
      </TabContext.Provider>
    </LearnProvider>,
  )
  const frame = rendered.container.querySelector("iframe")!
  const posted = vi.spyOn(frame.contentWindow!, "postMessage").mockImplementation(() => {})
  const fromFrame = (data: LearnFrameMessage | { type: "ready"; version: number }) => act(() => {
    window.dispatchEvent(new MessageEvent("message", { data: { channel: LEARN_EMBED_CHANNEL, ...data }, origin: FRAME_URL, source: frame.contentWindow }))
  })
  fromFrame({ type: "ready", version: LEARN_PROTOCOL_VERSION })
  const sent = (type: string) => posted.mock.calls.map(([envelope]) => envelope as { type: string; channel?: string })
    .filter((message) => message.type === type)
    .map((message) => Object.fromEntries(Object.entries(message).filter(([key]) => key !== "channel")))
  return { fromFrame, sent, onOpenItem }
}

const planButton = () => screen.queryByRole("button", { name: "Plan" })

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

it("shows the button only while the session has a plan", () => {
  const { fromFrame } = mountPlanButton({ kind: "session", sessionId: "s1" })
  expect(planButton()).toBeNull()

  fromFrame({ type: "session-plan", sessionId: "other", available: true })
  expect(planButton()).toBeNull()
  fromFrame({ type: "session-plan", sessionId: "s1", available: true })
  expect(planButton()).toBeInTheDocument()
  fromFrame({ type: "session-plan", sessionId: "s1", available: false })
  expect(planButton()).toBeNull()
})

it("opens LearnHarness's Plan popup on the tab's session", () => {
  const { fromFrame, sent, onOpenItem } = mountPlanButton({ kind: "session", sessionId: "s1" })
  fromFrame({ type: "view", sessionId: "s1" })
  fromFrame({ type: "session-plan", sessionId: "s1", available: true })

  fireEvent.click(planButton()!)

  expect(onOpenItem).toHaveBeenCalledWith({ kind: "learn", path: "", projectSlug: "p1", learn: { kind: "session", sessionId: "s1" } })
  expect(sent("open-panel")).toEqual([{ type: "open-panel", panel: "plan", sessionId: "s1" }])
})

it("is absent on /chat/new", () => {
  const { fromFrame } = mountPlanButton({ kind: "new" })
  fromFrame({ type: "view", sessionId: null })
  fromFrame({ type: "session-plan", sessionId: "s1", available: true })
  expect(planButton()).toBeNull()
})
