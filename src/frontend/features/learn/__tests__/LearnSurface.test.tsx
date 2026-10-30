import { act, cleanup, render } from "@testing-library/react"
import { afterEach, expect, it, vi } from "vitest"
import { TabContext, type Tab } from "@/contexts/TabContext"
import type { LearnSession, LearnView } from "@/lib/domain/types"
import type { LearnFrameAttachment } from "../LearnProvider"
import { LearnSurface } from "../LearnSurface"

const learn = vi.hoisted(() => ({
  learnUrl: "http://127.0.0.1:30181",
  frameStatus: "ready" as const,
  attachFrame: vi.fn<(attachment: LearnFrameAttachment) => () => void>(),
  reloadFrame: vi.fn(),
  sessions: { sessions: [] as LearnSession[], error: null, refresh: vi.fn(), claim: vi.fn() },
  sendToLearnFrame: vi.fn(),
}))
// A frame that never reported ready: no drive mode (`LearnSurfaceFrame.test.tsx` drives a fake frame).
vi.mock("../LearnProvider", () => ({ useLearn: () => learn, useLearnDrive: () => "waiting", useLearnFrameState: () => null, useLearnFrameEvent: () => {} }))

afterEach(() => {
  cleanup()
  vi.resetAllMocks()
  learn.sessions.sessions = []
})

function renderSurface(projectSlug: string | null, view: LearnView) {
  const setTitle = vi.fn()
  const onViewChange = vi.fn()
  learn.attachFrame.mockReturnValue(() => {})
  render(
    <TabContext.Provider value={{ tab: { projectSlug } as Tab, isActive: true, setModeLabel: vi.fn(), updateConfig: vi.fn(), setAppMode: vi.fn(), setWorkspaceItem: vi.fn(), setHistoryFile: vi.fn(), setTitle }}>
      <LearnSurface view={view} onViewChange={onViewChange} />
    </TabContext.Provider>,
  )
  const navigate = (next: LearnView) => act(() => learn.attachFrame.mock.lastCall![0].onViewChange(next))
  return { setTitle, onViewChange, navigate }
}

it("joins a session opened in a project's learn tab to that project", () => {
  const { onViewChange, navigate } = renderSurface("physics", { kind: "new" })

  navigate({ kind: "session", sessionId: "chat_new" })

  expect(onViewChange).toHaveBeenCalledWith({ kind: "session", sessionId: "chat_new" })
  expect(learn.sessions.claim).toHaveBeenCalledWith("chat_new", "physics")
})

it("joins nothing from a learn tab outside projects or on the way home", () => {
  const outside = renderSurface(null, { kind: "new" })
  outside.navigate({ kind: "session", sessionId: "chat_new" })
  cleanup()
  const project = renderSurface("physics", { kind: "session", sessionId: "chat_new" })
  project.navigate({ kind: "home" })

  expect(learn.sessions.claim).not.toHaveBeenCalled()
})

it("titles the tab with its session's title", () => {
  learn.sessions.sessions = [{ id: "chat_a", title: "Fourier series", kind: "teach", busy: false, project_slug: null }]
  const { setTitle } = renderSurface(null, { kind: "session", sessionId: "chat_a" })

  expect(setTitle).toHaveBeenLastCalledWith("Fourier series")
})
