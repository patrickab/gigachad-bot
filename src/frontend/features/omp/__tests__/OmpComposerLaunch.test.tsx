import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { act, cleanup, render } from "@testing-library/react"
import { OmpComposer, type ComposerLaunch } from "../OmpComposer"
import type { ComposerState } from "../composer"
import type { TabConfig } from "@/contexts/SettingsContext"
import { TabContext, type Tab } from "@/contexts/TabContext"

const omp = vi.hoisted(() => ({ composer: null as ComposerState | null, sendToComposer: vi.fn() }))

vi.mock("../OmpProvider", () => ({
  useOmp: () => ({
    composer: omp.composer, composerCommands: null, frameStatus: "ready", sendToComposer: omp.sendToComposer,
    onComposerEvent: () => () => {}, pendingSelect: null, answerSelect: () => {},
  }),
}))
vi.mock("@/features/chat/EnabledToolsContext", () => ({ useEnabledTools: () => ({ enabledTools: [], toggleTool: () => {} }) }))
vi.mock("@/contexts/SettingsContext", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/contexts/SettingsContext")>(),
  useSettings: () => ({ ocrModel: "" }),
  useNeovimSetting: () => ({ available: false, enabled: false, setPreferred: () => {} }),
}))

const NEW_SESSION: ComposerState = { type: "composer", sessionId: null, streaming: false, model: null, thinkingLevel: "auto", thinkingLevels: [], contextTokens: null }
const launch: ComposerLaunch = { text: "fix the build", attachments: [] }
const tab: Tab = { id: "t1", name: null, chatId: "c1", historyFile: null, title: null, config: {} as TabConfig, appMode: "omp", projectSlug: null, workspaceItem: null }
const shellCommands = { names: [], run: async () => false }

function renderLaunch(view: Parameters<typeof OmpComposer>[0]["view"], onLaunched = vi.fn()) {
  const props = { view, chatId: "c1", slug: null, shellCommands, launch, onLaunched }
  const tree = () => (
    <TabContext.Provider value={{ tab, isActive: true, setModeLabel: vi.fn(), updateConfig: vi.fn(), setAppMode: vi.fn(), setWorkspaceItem: vi.fn(), setHistoryFile: vi.fn(), setTitle: vi.fn() }}>
      <OmpComposer {...props} />
    </TabContext.Provider>
  )
  const mounted = render(tree())
  return { onLaunched, rerender: () => mounted.rerender(tree()) }
}

describe("OmpComposer launch", () => {
  beforeEach(() => { vi.useFakeTimers(); omp.composer = null; omp.sendToComposer.mockClear() })
  afterEach(() => { cleanup(); vi.useRealTimers() })

  it("waits for the new-session composer, then sends the first message once", async () => {
    const { onLaunched, rerender } = renderLaunch({ kind: "new", cwd: "/git/app" })
    await act(() => vi.advanceTimersByTimeAsync(1_000))
    expect(omp.sendToComposer).not.toHaveBeenCalled()

    omp.composer = NEW_SESSION
    rerender()
    await act(() => vi.advanceTimersByTimeAsync(1_000))

    expect(omp.sendToComposer).toHaveBeenCalledTimes(1)
    expect(omp.sendToComposer).toHaveBeenCalledWith(expect.objectContaining({ type: "composer-send", sessionId: null, text: "fix the build" }))
    expect(onLaunched).toHaveBeenCalledTimes(1)
  })

  it("never sends into a session's composer", async () => {
    omp.composer = { ...NEW_SESSION, sessionId: "s1" }
    renderLaunch({ kind: "session", sessionId: "s1" })
    await act(() => vi.advanceTimersByTimeAsync(1_000))
    expect(omp.sendToComposer).not.toHaveBeenCalled()
  })
})
