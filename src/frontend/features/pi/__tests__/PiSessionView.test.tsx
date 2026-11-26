import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { fetchPiFiles } from "@/lib/backend/api"
import type { ModelsResponse } from "@/lib/domain/types"
import { PiSessionView, withLead } from "../PiTab"
import { PiPanelBar } from "../PiTopBar"
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

  it("stages an insertion into the composer and completes @ in the chat's folder", async () => {
    const onInserted = vi.fn()
    render(<PiSessionView sessionId="d1" cwd="/work/markdown" files={{ cwd: "/work/markdown" }} renderHeader={() => null} insertion={{ id: 1, text: "notes.md:2-3\n> a quote" }} onInserted={onInserted} />)

    await waitFor(() => expect(screen.getByRole("textbox")).toHaveValue("notes.md:2-3\n> a quote"))
    expect(onInserted).toHaveBeenCalledOnce()
    expect(fetchPiFiles).toHaveBeenCalledWith({ cwd: "/work/markdown" })
  })

  it("shows the panel bar like a chat's top bar: thinking, the chats' model dropdown, session, context readout", () => {
    const models: ModelsResponse = {
      ollama: [],
      providers: [{ label: "Pi", provider_id: "litellm_proxy", models: ["anthropic/claude-haiku-4-5"], source: "pi" }],
      defaults: { default_model: "", small_model: "", vision_model: "", sandbox_model: "" },
      tab_order: ["Pi"],
      unavailable: [],
    }
    const modelProps = { models, onProvidersChange: async () => {}, onDefaultsChange: async () => {}, placement: "below" as const }
    render(<PiSessionView sessionId="d1" cwd="/work/markdown" files={{ cwd: "/work/markdown" }} renderHeader={(header) => <PiPanelBar header={header} modelProps={modelProps} actions={<button type="button">Sessions</button>} />} />)

    expect(screen.getByText("Notes chat")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Model: anthropic/claude-haiku-4-5" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Reasoning: medium" })).toBeInTheDocument()
    expect(screen.getByLabelText("Cost 0.12$ (0.00$), context 20k tokens, 10% of 200k")).toHaveTextContent("0.12$ (0.00$)·20k (10%)")
    expect(screen.getByRole("button", { name: "Sessions" })).toBeInTheDocument()
  })
})

describe("withLead", () => {
  it("leads a prompt but never a Pi command", () => {
    expect(withLead("hi", "File: /a.md")).toBe("File: /a.md\n\nhi")
    expect(withLead("/skill:tex fix", "File: /a.md")).toBe("/skill:tex fix")
    expect(withLead("hi", undefined)).toBe("hi")
  })
})
