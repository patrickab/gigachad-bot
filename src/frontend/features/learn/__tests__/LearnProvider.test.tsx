import { act, cleanup, render } from "@testing-library/react"
import { useEffect, type ReactNode } from "react"
import { afterEach, expect, it, vi } from "vitest"
import { LEARN_EMBED_CHANNEL, LEARN_PROTOCOL_VERSION } from "@gigachad/chat-ui/learn-protocol"
import type { LearnView } from "@/lib/domain/types"
import { LearnProvider, useLearn } from "../LearnProvider"

const FRAME_URL = "http://127.0.0.1:30181"
const settings = vi.hoisted(() => ({ hostAccess: false, learnharnessUrl: null as string | null }))
vi.mock("@/contexts/SettingsContext", () => ({ useSettings: () => settings }))
vi.mock("@/hooks/useThemeName", () => ({ useThemeName: () => "light" }))
vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} })
const sessions = vi.hoisted(() => ({
  sessions: [] as import("@/lib/domain/types").LearnSession[],
  retitle: vi.fn<(sessionId: string, title: string) => void>(),
  refresh: vi.fn(async () => {}),
}))
vi.mock("../useLearnSessions", () => ({ useLearnSessions: () => sessions }))

function Tab({ view, onViewChange }: { view: LearnView; onViewChange: (view: LearnView) => void }) {
  const { attachFrame } = useLearn()
  useEffect(() => attachFrame({ slot: document.body, view, onViewChange }), [attachFrame, view, onViewChange])
  return null
}

/** Render `children` beside an attached learn tab and return the frame's postMessage spy and a way to post from the frame. */
function mountFrame(children: ReactNode, onViewChange = vi.fn(), onOpenItem = vi.fn()) {
  settings.hostAccess = true
  settings.learnharnessUrl = FRAME_URL
  const view = render(<LearnProvider onOpenItem={onOpenItem}><Tab view={{ kind: "home" }} onViewChange={onViewChange} />{children}</LearnProvider>)
  const frame = view.container.querySelector("iframe")!
  const posted = vi.spyOn(frame.contentWindow!, "postMessage").mockImplementation(() => {})
  const fromFrame = (data: object, origin = FRAME_URL, source: Window | null = frame.contentWindow) => act(() => {
    window.dispatchEvent(new MessageEvent("message", { data: { channel: LEARN_EMBED_CHANNEL, ...data }, origin, source }))
  })
  return { view, frame, posted, fromFrame }
}

afterEach(() => {
  cleanup()
  settings.hostAccess = false
  settings.learnharnessUrl = null
  sessions.sessions = []
  sessions.retitle.mockClear()
  sessions.refresh.mockClear()
})

it("mounts no frame for logins without host access", () => {
  settings.learnharnessUrl = FRAME_URL
  const view = render(<LearnProvider onOpenItem={vi.fn()}><Tab view={{ kind: "home" }} onViewChange={vi.fn()} /></LearnProvider>)
  expect(view.container.querySelector("iframe")).toBeNull()
})

it("greets the frame from its own window and origin only, and hands its navigation to the attached tab", () => {
  const onViewChange = vi.fn()
  const { frame, posted, fromFrame } = mountFrame(null, onViewChange)
  expect(frame.getAttribute("src")).toBe(FRAME_URL)

  fromFrame({ type: "ready", version: LEARN_PROTOCOL_VERSION }, "http://elsewhere.test")
  fromFrame({ type: "ready", version: LEARN_PROTOCOL_VERSION }, FRAME_URL, window)
  expect(posted).not.toHaveBeenCalled()

  fromFrame({ type: "ready", version: LEARN_PROTOCOL_VERSION })
  expect(posted).toHaveBeenNthCalledWith(1, { channel: LEARN_EMBED_CHANNEL, type: "hello", theme: "light", version: LEARN_PROTOCOL_VERSION }, FRAME_URL)
  expect(posted).toHaveBeenNthCalledWith(2, { channel: LEARN_EMBED_CHANNEL, type: "open-home" }, FRAME_URL)
  expect(frame.style.opacity).toBe("1")

  fromFrame({ type: "view", sessionId: null })
  fromFrame({ type: "view", sessionId: "s1" })
  expect(onViewChange).toHaveBeenCalledWith({ kind: "session", sessionId: "s1" })
})
