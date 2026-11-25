import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { ApiError, callFactory, fetchPiCatalog, fetchPiSessions, openPiSession, restorePiSession, type PiSessionInfo } from "@/lib/backend/api"
import type { ModelsResponse, PiCatalog } from "@/lib/domain/types"
import { PiTab } from "../PiTab"
import { EMPTY_TRANSCRIPT, type Transcript } from "../transcript"

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
  fetchPiCatalog: vi.fn(),
  openPiSession: vi.fn(),
  restorePiSession: vi.fn(),
  callFactory: vi.fn(async () => ({ waves: [] })),
}))
vi.mock("../sessionStore", () => ({ refreshPiSessions: vi.fn() }))
vi.mock("../piSocket", async (importOriginal) => ({ ...await importOriginal<typeof import("../piSocket")>(), observePiSocket: () => () => {} }))
vi.mock("../PiTranscript", () => ({
  PiTranscript: ({ onBranch }: { onBranch?: (entryId: string) => void }) => onBranch ? <button type="button" onClick={() => onBranch("entry-7")}>Branch</button> : null,
}))

const { send, session } = vi.hoisted(() => ({ send: vi.fn(async () => {}), session: { transcript: null as Transcript | null } }))
// Stable identities, like the real hook
const NO_WIDGETS = {}
const inspectSubagent = vi.fn()
vi.mock("../usePiSession", () => ({
  usePiSession: () => ({
    transcript: session.transcript ?? EMPTY_TRANSCRIPT,
    status: { kind: "ready" },
    state: { model: null, thinkingLevel: "medium", isStreaming: false, sessionId: "s1", sessionFile: "/sessions/s1.jsonl" },
    btw: { threads: [], followUp: () => {}, cancel: () => {}, dismiss: () => {} }, commands: [], dialogs: [], notices: [], statusLines: {}, widgets: NO_WIDGETS, inspectSubagent, editorText: null, models: [], thinkingLevels: [], stats: null,
    send, abort: async () => [], runBash: async () => {}, compact: async () => {}, setModel: async () => {}, setThinking: async () => {},
    answerDialog: () => {}, notify: () => {},
  }),
}))

const CATALOG: PiCatalog = {
  installed: true, online: true, gateway: "", provider_id: "litellm_proxy", error: null,
  providers: [{
    id: "anthropic",
    label: "Anthropic",
    models: [
      { id: "anthropic/claude-haiku-4-5", name: "Claude Haiku 4.5", vision: true, context_window: 200000, thinking_levels: ["off", "minimal", "low", "medium", "high"] },
      { id: "anthropic/claude-3-haiku", name: "Claude Haiku 3", vision: false, context_window: 200000, thinking_levels: [] },
    ],
  }, {
    id: "openrouter",
    label: "OpenRouter",
    models: [{ id: "openrouter/acme/model-x", name: "acme/model-x", vision: false, context_window: 128000, thinking_levels: [] }],
  }],
}

// The user's catalog: a Pi row pick, a model typed under OpenRouter, and one the gateway does not serve
const MODELS: ModelsResponse = {
  ollama: [],
  providers: [
    { label: "Pi", provider_id: "litellm_proxy", models: ["anthropic/claude-haiku-4-5"], source: "pi" },
    { label: "OpenRouter", provider_id: "openrouter", models: ["acme/model-x", "acme/gone"] },
  ],
  defaults: { default_model: "", small_model: "", vision_model: "", sandbox_model: "" },
  tab_order: ["Pi", "OpenRouter"],
  unavailable: ["openrouter/acme/gone"],
}
const MODEL_PROPS = { models: MODELS, onProvidersChange: async () => {}, onDefaultsChange: async () => {}, placement: "below" as const }

function sessionInfo(sessionId: string): PiSessionInfo {
  return {
    sessionId, sessionPath: `/sessions/${sessionId}.jsonl`, cwd: "/home/u/git/repo", machine: "local", createdAt: "", modifiedAt: "",
    firstMessage: "", messageCount: 0, tokens: 0, live: null, pinned: false, archived: false,
  }
}

function sendText(text: string) {
  fireEvent.change(screen.getByRole("textbox"), { target: { value: text } })
  fireEvent.keyDown(screen.getByRole("textbox"), { key: "Enter" })
}

describe("PiTab", () => {
  let stored: Map<string, string>
  beforeEach(() => {
    stored = new Map<string, string>()
    vi.stubGlobal("localStorage", {
      getItem: (key: string) => stored.get(key) ?? null,
      setItem: (key: string, value: string) => { stored.set(key, value) },
      removeItem: (key: string) => { stored.delete(key) },
    })
    vi.mocked(fetchPiCatalog).mockResolvedValue(CATALOG)
    session.transcript = null
  })
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
    vi.unstubAllGlobals()
  })

  it("starts a new session at thinking medium on Pi's default model, then sends the first message", async () => {
    vi.mocked(openPiSession).mockResolvedValue(sessionInfo("s1"))
    const onViewChange = vi.fn()
    const { rerender } = render(<PiTab view={{ repo: "repo", sessionId: null }} onViewChange={onViewChange} modelProps={MODEL_PROPS} />)
    sendText("hello")

    await waitFor(() => expect(onViewChange).toHaveBeenCalledWith({ repo: "repo", sessionId: "s1" }))
    expect(openPiSession).toHaveBeenCalledWith("repo", null, { thinking: "medium" })
    rerender(<PiTab view={{ repo: "repo", sessionId: "s1" }} onViewChange={onViewChange} modelProps={MODEL_PROPS} />)
    await waitFor(() => expect(send).toHaveBeenCalledWith("hello", [], "steer"))
  })

  it("starts a new session on the catalog model and thinking level picked in the empty tab's top bar", async () => {
    vi.mocked(openPiSession).mockResolvedValue(sessionInfo("s1"))
    const topBar = document.createElement("div")
    const slots = { thinking: topBar.appendChild(document.createElement("div")), model: topBar.appendChild(document.createElement("div")), usage: null }
    document.body.append(topBar)
    render(<PiTab view={{ repo: "repo", sessionId: null }} onViewChange={vi.fn()} modelProps={MODEL_PROPS} slots={slots} />)

    fireEvent.click(await screen.findByRole("button", { name: "Model: Select model" }))
    fireEvent.click(screen.getByRole("menuitemradio", { name: "anthropic/claude-haiku-4-5" }))
    await waitFor(() => expect(screen.getByRole("button", { name: "Model: anthropic/claude-haiku-4-5" })).toBeInTheDocument())
    fireEvent.click(screen.getByRole("button", { name: "Reasoning: medium" }))
    fireEvent.click(screen.getByRole("menuitemradio", { name: "high" }))
    sendText("hello")

    await waitFor(() => expect(openPiSession).toHaveBeenCalledWith("repo", null, { model: { provider: "anthropic", id: "claude-haiku-4-5" }, thinking: "high" }))
    topBar.remove()
  })

  it("offers only the catalog models Pi serves and starts on a typed OpenRouter model", async () => {
    vi.mocked(openPiSession).mockResolvedValue(sessionInfo("s1"))
    const topBar = document.createElement("div")
    const slots = { thinking: topBar.appendChild(document.createElement("div")), model: topBar.appendChild(document.createElement("div")), usage: null }
    document.body.append(topBar)
    render(<PiTab view={{ repo: "repo", sessionId: null }} onViewChange={vi.fn()} modelProps={MODEL_PROPS} slots={slots} />)

    fireEvent.click(await screen.findByRole("button", { name: "Model: Select model" }))
    fireEvent.click(screen.getByRole("button", { name: "OpenRouter" }))
    expect(screen.queryByRole("menuitemradio", { name: "acme/gone" })).toBeNull()
    fireEvent.click(screen.getByRole("menuitemradio", { name: "acme/model-x" }))
    sendText("hello")

    await waitFor(() => expect(openPiSession).toHaveBeenCalledWith("repo", null, { model: { provider: "openrouter", id: "acme/model-x" }, thinking: "medium" }))
    topBar.remove()
  })

  it("keeps an empty tab's draft per repo", () => {
    render(<PiTab view={{ repo: "repo", sessionId: null }} onViewChange={vi.fn()} modelProps={MODEL_PROPS} />)
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "half a thought" } })
    expect(stored.get("gigachad:draft:pi:new:repo")).toBe("half a thought")
  })

  it("restores an archived session before reopening it", async () => {
    vi.mocked(fetchPiSessions).mockImplementation(async (_repo, options) => (options?.archived ? [{ ...sessionInfo("s1"), archived: true }] : []))
    vi.mocked(restorePiSession).mockResolvedValue(sessionInfo("s1"))
    vi.mocked(openPiSession).mockResolvedValue(sessionInfo("s1"))
    render(<PiTab view={{ repo: "repo", sessionId: "s1" }} onViewChange={vi.fn()} modelProps={MODEL_PROPS} />)

    await waitFor(() => expect(openPiSession).toHaveBeenCalledWith("repo", { sessionPath: "/sessions/s1.jsonl" }))
    expect(restorePiSession).toHaveBeenCalledWith("s1")
  })

  it("says a session neither listed nor archived no longer exists", async () => {
    render(<PiTab view={{ repo: "repo", sessionId: "gone" }} onViewChange={vi.fn()} modelProps={MODEL_PROPS} />)
    expect(await screen.findByRole("alert")).toHaveTextContent("This session no longer exists")
    expect(restorePiSession).not.toHaveBeenCalled()
  })

  it("explains a missing folder", async () => {
    vi.mocked(openPiSession).mockRejectedValue(new ApiError("Unknown repository notes", 404))
    render(<PiTab view={{ repo: "notes", sessionId: null }} onViewChange={vi.fn()} modelProps={MODEL_PROPS} />)
    sendText("hello")
    expect(await screen.findByText("notes is not a folder")).toHaveClass("text-danger")
  })

  it("opens a turn's branch as a fork of the session in the same tab", async () => {
    vi.mocked(fetchPiSessions).mockResolvedValue([sessionInfo("s1")])
    vi.mocked(openPiSession).mockResolvedValueOnce(sessionInfo("s1")).mockResolvedValueOnce(sessionInfo("s2"))
    const onViewChange = vi.fn()
    render(<PiTab view={{ repo: "repo", sessionId: "s1" }} onViewChange={onViewChange} modelProps={MODEL_PROPS} />)
    fireEvent.click(await screen.findByRole("button", { name: "Branch" }))

    await waitFor(() => expect(onViewChange).toHaveBeenCalledWith({ repo: "repo", sessionId: "s2" }))
    expect(openPiSession).toHaveBeenCalledWith("repo", { forkOf: { sessionPath: "/sessions/s1.jsonl", entryId: "entry-7" } })
  })

  it("reads the session's waves for its absolute cwd when the view names the checkout", async () => {
    vi.mocked(fetchPiSessions).mockResolvedValue([sessionInfo("s1")])
    vi.mocked(openPiSession).mockResolvedValue(sessionInfo("s1"))
    render(<PiTab view={{ repo: "repo", sessionId: "s1" }} onViewChange={vi.fn()} modelProps={MODEL_PROPS} onOpenSession={vi.fn()} />)

    await waitFor(() => expect(callFactory).toHaveBeenCalledWith("waves.list", { repo: "/home/u/git/repo" }))
    expect(callFactory).not.toHaveBeenCalledWith("waves.list", { repo: "repo" })
  })
})
