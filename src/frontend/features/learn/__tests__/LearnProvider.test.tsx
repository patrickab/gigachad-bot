import { act, cleanup, render } from "@testing-library/react"
import { useEffect, type ReactNode } from "react"
import { afterEach, expect, it, vi } from "vitest"
import { LEARN_EMBED_CHANNEL, LEARN_PROTOCOL_VERSION } from "@gigachad/chat-ui/learn-protocol"
import type { LearnView } from "@/lib/domain/types"
import { LearnProvider, useLearn, useLearnDrive, useLearnFrameEvent, useLearnFrameState } from "../LearnProvider"
import type { LearnDriveMessage } from "../bridge"

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

const meta = (sessionId: string, title: string) =>
  ({ type: "session-meta", sessionId, title, kind: "teach", phase: null, status: null, updatedAt: "2026-10-29T10:00:00.000Z" }) as const

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

it("exposes the frame's state, events and drive mode, and sends drive messages only to a version-matched frame", () => {
  const seen: { drive: string; modal: boolean | null; titles: string[] }[] = []
  const opened: (string | null)[] = []
  let send: (message: LearnDriveMessage) => boolean = () => false
  function Consumer() {
    const drive = useLearnDrive()
    const modal = useLearnFrameState("modal")
    const meta = useLearnFrameState("session-meta")
    const { sendToLearnFrame } = useLearn()
    useEffect(() => {
      send = sendToLearnFrame
    }, [sendToLearnFrame])
    useLearnFrameEvent("model-open", ({ sessionId }) => opened.push(sessionId))
    seen.push({ drive, modal: modal?.open ?? null, titles: Object.values(meta).map(({ title }) => title) })
    return null
  }
  const { posted, fromFrame } = mountFrame(<Consumer />)
  expect(seen.at(-1)).toEqual({ drive: "waiting", modal: null, titles: [] })

  fromFrame({ type: "ready" })
  expect(seen.at(-1)?.drive).toBe("mismatch")
  posted.mockClear()
  expect(send({ type: "open-settings", section: null })).toBe(false)
  expect(posted).not.toHaveBeenCalled()

  fromFrame({ type: "ready", version: LEARN_PROTOCOL_VERSION })
  fromFrame({ type: "modal", open: true })
  fromFrame(meta("s1", "Limits"))
  fromFrame({ ...meta("s1", "Limits"), title: 3 })
  fromFrame({ type: "model-open", sessionId: "s1" })
  expect(seen.at(-1)).toEqual({ drive: "matched", modal: true, titles: ["Limits"] })
  expect(opened).toEqual(["s1"])

  posted.mockClear()
  expect(send({ type: "open-settings", section: null })).toBe(true)
  expect(posted).toHaveBeenCalledWith({ channel: LEARN_EMBED_CHANNEL, type: "open-settings", section: null }, FRAME_URL)
})

it("adopts each title the frame reports once, and reads the list for a session it lacks", () => {
  sessions.sessions = [{ id: "s1", title: "Limits", kind: "teach", busy: false, project_slug: null }]
  const { fromFrame } = mountFrame(null)
  fromFrame({ type: "ready", version: LEARN_PROTOCOL_VERSION })
  fromFrame(meta("s1", "Limits, renamed in the frame"))
  fromFrame(meta("s2", "Started in the frame"))
  fromFrame({ type: "modal", open: false })
  expect(sessions.retitle.mock.calls).toEqual([["s1", "Limits, renamed in the frame"]])
  expect(sessions.refresh).toHaveBeenCalledOnce()
})

it("shows upstream's UI in the frame once it is driven and on the asked session", () => {
  let showInFrame: ReturnType<typeof useLearn>["showInFrame"] = () => {}
  function Consumer() {
    const learn = useLearn()
    useEffect(() => {
      showInFrame = learn.showInFrame
    }, [learn.showInFrame])
    return null
  }
  const onOpenItem = vi.fn()
  const { posted, fromFrame } = mountFrame(<Consumer />, vi.fn(), onOpenItem)
  const coverage = { type: "open-panel", panel: "coverage", sessionId: "s1" } as const
  const sent = () => posted.mock.calls.map(([message]) => message).filter((message) => message.type === "open-panel" || message.type === "open-settings")

  act(() => showInFrame(coverage, { id: "s1", project_slug: "physics" }))
  expect(onOpenItem).toHaveBeenLastCalledWith({ kind: "learn", path: "", projectSlug: "physics", learn: { kind: "session", sessionId: "s1" } })
  fromFrame({ type: "ready", version: LEARN_PROTOCOL_VERSION })
  fromFrame({ type: "view", sessionId: null })
  expect(sent()).toEqual([])
  fromFrame({ type: "view", sessionId: "s1" })
  expect(sent()).toEqual([{ channel: LEARN_EMBED_CHANNEL, ...coverage }])

  // An app panel or the settings need no session: a learn tab is in front, they go at once.
  act(() => showInFrame({ type: "open-settings", section: null }))
  expect(onOpenItem).toHaveBeenCalledOnce()
  expect(sent()).toHaveLength(2)

  // A frame at another version keeps upstream's own chrome: nothing is sent.
  fromFrame({ type: "ready", version: LEARN_PROTOCOL_VERSION + 1 })
  posted.mockClear()
  act(() => showInFrame({ type: "open-panel", panel: "activity", sessionId: null }))
  expect(sent()).toEqual([])
})
