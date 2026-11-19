import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { fetchPiSessions, openPiSession, type PiSessionInfo } from "@/lib/backend/api"
import { PiTab } from "../PiTab"
import { EMPTY_TRANSCRIPT } from "../transcript"

vi.mock("@/features/chat/EnabledToolsContext", () => ({ useEnabledTools: () => ({ enabledTools: [], toggleTool: () => {} }) }))
vi.mock("@/contexts/SettingsContext", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/contexts/SettingsContext")>(),
  useSettings: () => ({ ocrModel: "" }),
  useNeovimSetting: () => ({ available: false, enabled: false, setPreferred: () => {} }),
}))
vi.mock("@/features/editor/FilePanel", () => ({ useOpenFile: () => () => {} }))
vi.mock("@/lib/backend/api", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/lib/backend/api")>(),
  fetchPiFiles: vi.fn(async () => []),
  fetchPiSessions: vi.fn(async () => []),
  openPiSession: vi.fn(),
}))
vi.mock("../PiTranscript", () => ({
  PiTranscript: ({ onBranch }: { onBranch?: (entryId: string) => void }) => onBranch ? <button type="button" onClick={() => onBranch("entry-7")}>Branch</button> : null,
}))

const { send } = vi.hoisted(() => ({ send: vi.fn(async () => {}) }))
vi.mock("../usePiSession", () => ({
  usePiSession: () => ({
    transcript: EMPTY_TRANSCRIPT,
    status: { kind: "ready" },
    state: { model: null, thinkingLevel: "medium", isStreaming: false, sessionId: "s1", sessionFile: "/sessions/s1.jsonl" },
    commands: [], dialogs: [], notices: [], statusLines: {}, editorText: null, models: [], thinkingLevels: [], stats: null,
    send, abort: async () => [], runBash: async () => {}, compact: async () => {}, setModel: async () => {}, setThinking: async () => {},
    answerDialog: () => {}, notify: () => {},
  }),
}))

function sessionInfo(sessionId: string): PiSessionInfo {
  return {
    sessionId, sessionPath: `/sessions/${sessionId}.jsonl`, cwd: "/home/u/git/repo", machine: "local", createdAt: "", modifiedAt: "",
    firstMessage: "", messageCount: 0, live: null, pinned: false, archived: false,
  }
}

describe("PiTab", () => {
  beforeEach(() => {
    const stored = new Map<string, string>()
    vi.stubGlobal("localStorage", {
      getItem: (key: string) => stored.get(key) ?? null,
      setItem: (key: string, value: string) => { stored.set(key, value) },
      removeItem: (key: string) => { stored.delete(key) },
    })
  })
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
    vi.unstubAllGlobals()
  })

  it("starts a new session at thinking medium on Pi's default model, then sends the first message", async () => {
    vi.mocked(openPiSession).mockResolvedValue(sessionInfo("s1"))
    const onViewChange = vi.fn()
    const { rerender } = render(<PiTab view={{ repo: "repo", sessionId: null }} tabId="t1" onViewChange={onViewChange} />)
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "hello" } })
    fireEvent.keyDown(screen.getByRole("textbox"), { key: "Enter" })

    await waitFor(() => expect(onViewChange).toHaveBeenCalledWith({ repo: "repo", sessionId: "s1" }))
    expect(openPiSession).toHaveBeenCalledWith("repo", null, { thinking: "medium" })
    rerender(<PiTab view={{ repo: "repo", sessionId: "s1" }} tabId="t1" onViewChange={onViewChange} />)
    await waitFor(() => expect(send).toHaveBeenCalledWith("hello", [], "steer"))
  })

  it("opens a turn's branch as a fork of the session in the same tab", async () => {
    vi.mocked(fetchPiSessions).mockResolvedValue([sessionInfo("s1")])
    vi.mocked(openPiSession).mockResolvedValueOnce(sessionInfo("s1")).mockResolvedValueOnce(sessionInfo("s2"))
    const onViewChange = vi.fn()
    render(<PiTab view={{ repo: "repo", sessionId: "s1" }} tabId="t1" onViewChange={onViewChange} />)
    fireEvent.click(await screen.findByRole("button", { name: "Branch" }))

    await waitFor(() => expect(onViewChange).toHaveBeenCalledWith({ repo: "repo", sessionId: "s2" }))
    expect(openPiSession).toHaveBeenCalledWith("repo", { forkOf: { sessionPath: "/sessions/s1.jsonl", entryId: "entry-7" } })
  })
})
