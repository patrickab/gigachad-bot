import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { PiSessionView } from "../PiTab"
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
}))
vi.mock("../PiTranscript", () => ({ PiTranscript: () => null }))

const { send, runBash } = vi.hoisted(() => ({ send: vi.fn(async () => {}), runBash: vi.fn(async () => {}) }))
// Stable identities, like the real hook
const NO_WIDGETS = {}
const inspectSubagent = vi.fn()
vi.mock("../usePiSession", () => ({
  usePiTranscript: (_id: string, select = (t: unknown) => t) => select(EMPTY_TRANSCRIPT),
  usePiSession: () => ({
    transcript: EMPTY_TRANSCRIPT,
    status: { kind: "ready" },
    state: { model: { provider: "anthropic", id: "claude-haiku-4-5", name: "Claude Haiku 4.5", reasoning: true }, thinkingLevel: "medium", isStreaming: false, sessionId: "d1", sessionName: "Notes chat" },
    btw: { threads: [], followUp: () => {}, cancel: () => {}, dismiss: () => {} }, commands: [], dialogs: [], notices: [], statusLines: {}, widgets: NO_WIDGETS, inspectSubagent, editorText: null,
    models: [{ provider: "anthropic", id: "claude-haiku-4-5", name: "Claude Haiku 4.5", reasoning: true }],
    thinkingLevels: ["off", "low", "medium", "high"],
    stats: { cost: 0.12, contextUsage: { tokens: 20_000, contextWindow: 200_000, percent: 10 } },
    send, abort: async () => [], runBash, compact: async () => {}, setModel: async () => {}, setThinking: async () => {},
    answerDialog: () => {}, notify: () => {},
  }),
}))

const PDF_LEAD = "Text: /work/pdf/.text/paper.txt"

function sendText(text: string) {
  fireEvent.change(screen.getByRole("textbox"), { target: { value: text } })
  fireEvent.keyDown(screen.getByRole("textbox"), { key: "Enter" })
}

describe("PiSessionView in a document panel", () => {
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

  it("leads the first prompt with `lead` once, and keeps it off commands", async () => {
    const onLeadSent = vi.fn()
    render(<PiSessionView sessionId="d1" cwd="/work/pdf" files={{ cwd: "/work/pdf" }} renderHeader={() => null} lead={PDF_LEAD} onLeadSent={onLeadSent} />)

    sendText("!ls")
    await waitFor(() => expect(runBash).toHaveBeenCalledWith("ls", false))
    expect(onLeadSent).not.toHaveBeenCalled()

    sendText("summarize the paper")
    await waitFor(() => expect(send).toHaveBeenCalledWith(`${PDF_LEAD}\n\nsummarize the paper`, [], "steer"))
    expect(onLeadSent).toHaveBeenCalledOnce()

    // The owner has not cleared `lead` yet: the next prompt still goes without it
    sendText("and the method?")
    await waitFor(() => expect(send).toHaveBeenLastCalledWith("and the method?", [], "steer"))
    expect(onLeadSent).toHaveBeenCalledOnce()
  })
})
