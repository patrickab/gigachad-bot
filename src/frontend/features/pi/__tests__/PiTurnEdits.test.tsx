import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { ApiError, deletePiTurn } from "@/lib/backend/api"
import { PiSessionView } from "../PiTab"
import { EMPTY_TRANSCRIPT, type Transcript } from "../transcript"
import type { AgentMessage } from "../types"

/** A Pi session's turn delete and edit, wired as a Pi tab wires them: the host's route, then the session's prompt. */

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
  fetchPiSessionMatches: vi.fn(async () => []),
  deletePiTurn: vi.fn(async () => ({ removedEntries: 2, notRemoved: [] as string[] })),
}))

const session = vi.hoisted(() => ({
  transcript: null as unknown,
  send: vi.fn(async (..._args: unknown[]) => true),
  notify: vi.fn(),
}))
const NO_WIDGETS = {}
const STATE = { model: null, thinkingLevel: "medium", isStreaming: false, sessionId: "s-1", sessionName: "Turns" }
const BTW = { threads: [], followUp: () => {}, cancel: () => {}, dismiss: () => {} }
const NONE: never[] = []
vi.mock("../usePiSession", () => ({
  usePiTranscript: (_id: string, select = (t: unknown) => t) => select(session.transcript),
  usePiSession: () => ({
    transcript: session.transcript,
    status: { kind: "ready" },
    state: STATE,
    btw: BTW, commands: NONE, dialogs: NONE, notices: NONE, statusLines: NO_WIDGETS, widgets: NO_WIDGETS, inspectSubagent: () => {}, editorText: null,
    models: NONE, thinkingLevels: NONE, stats: null,
    send: session.send, abort: async () => [], runBash: async () => {}, compact: async () => {}, setModel: async () => {}, setThinking: async () => {},
    answerDialog: () => {}, notify: session.notify,
  }),
}))

const IMAGE = { type: "image" as const, data: "aGk=", mimeType: "image/png" }

function history(turnCount: number): AgentMessage[] {
  return Array.from({ length: turnCount }, (_, index): AgentMessage[] => [
    { role: "user", content: index === 0 ? [{ type: "text", text: "Prompt 0" }, IMAGE] : `Prompt ${index}`, timestamp: index * 10 },
    { role: "assistant", content: [{ type: "text", text: `Answer ${index}` }], provider: "anthropic", model: "claude", stopReason: "stop", timestamp: index * 10 + 1 },
  ]).flat()
}

function useTranscript(messages: AgentMessage[], extra: Partial<Transcript> = {}) {
  session.transcript = { ...EMPTY_TRANSCRIPT, messages, entryIds: messages.map((_, index) => `e${index}`), ...extra }
}

function renderSession() {
  render(<PiSessionView sessionId="s-1" cwd="/repo" files={{ repo: "repo" }} renderHeader={() => null} />)
}

/** The question action `name` of the turn whose prompt is `prompt`, opened first. */
function questionAction(prompt: string, name: string) {
  const header = screen.queryByRole("button", { name: prompt })
  if (header) fireEvent.click(header)
  const turn = screen.getAllByRole("group", { name: "Question actions" }).find((group) => group.closest("[data-turn-key]")?.textContent?.includes(prompt))!
  return within(turn).getByRole("button", { name })
}

/** Confirm the open dialog with Enter on its button `name`. */
function confirm(name: string) {
  fireEvent.keyDown(screen.getByRole("button", { name }), { key: "Enter" })
}

function edit(prompt: string, text: string) {
  fireEvent.click(questionAction(prompt, "Edit message"))
  const editor = screen.getByRole("textbox", { name: "Edit question" })
  fireEvent.change(editor, { target: { value: text } })
  fireEvent.keyDown(editor, { key: "Enter" })
}

beforeAll(() => { Element.prototype.scrollIntoView = () => {} })

describe("a Pi turn's delete and edit", () => {
  beforeEach(() => {
    vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} })
    const stored = new Map<string, string>()
    vi.stubGlobal("localStorage", {
      getItem: (key: string) => stored.get(key) ?? null,
      setItem: (key: string, value: string) => { stored.set(key, value) },
      removeItem: (key: string) => { stored.delete(key) },
    })
    useTranscript(history(2))
  })
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
    vi.unstubAllGlobals()
  })

  it("deletes the turn through its prompt's entry once confirmed, removing nothing itself", async () => {
    renderSession()
    fireEvent.click(questionAction("Prompt 1", "Delete turn"))
    expect(deletePiTurn).not.toHaveBeenCalled()
    confirm("Delete")
    await waitFor(() => expect(deletePiTurn).toHaveBeenCalledWith("s-1", "e2", "turn"))
    expect(screen.getByText("Prompt 1")).toBeInTheDocument()
    expect(session.send).not.toHaveBeenCalled()
    expect(session.notify).not.toHaveBeenCalled()
  })

  it("shows one notice when a delete left files on disk", async () => {
    vi.mocked(deletePiTurn).mockResolvedValueOnce({ removedEntries: 2, notRemoved: [".pi/pastes/a.txt", "f00d"] })
    renderSession()
    fireEvent.click(questionAction("Prompt 1", "Delete turn"))
    confirm("Delete")
    await waitFor(() => expect(session.notify).toHaveBeenCalledOnce())
    expect(session.notify).toHaveBeenCalledWith("The turn is deleted, but 2 files of it could not be removed from disk: .pi/pastes/a.txt, f00d", "warning")
  })

  it("edits by deleting through the end, then prompting once with the turn's images", async () => {
    renderSession()
    edit("Prompt 0", "Prompt zero, again")
    // One later turn goes with it: asked first
    expect(deletePiTurn).not.toHaveBeenCalled()
    confirm("Edit")
    await waitFor(() => expect(session.send).toHaveBeenCalledOnce())
    expect(deletePiTurn).toHaveBeenCalledWith("s-1", "e0", "end")
    expect(session.send).toHaveBeenCalledWith("Prompt zero, again", [IMAGE])
    expect(vi.mocked(deletePiTurn).mock.invocationCallOrder[0]).toBeLessThan(session.send.mock.invocationCallOrder[0])
  })

  it("puts the edited text into the composer when the send fails", async () => {
    session.send.mockResolvedValueOnce(false)
    renderSession()
    edit("Prompt 1", "Prompt one, again")
    await waitFor(() => expect(screen.getByRole("textbox")).toHaveValue("Prompt one, again"))
  })

  it.each([
    ["running", "Wait for the run to finish"],
    ["compacted", "A compaction already summarized this turn"],
    ["locked", "The session file is in use"],
    ["changed", "The session file is in use"],
  ])("shows the refusal %s as a sentence and keeps the turn", async (code, sentence) => {
    vi.mocked(deletePiTurn).mockRejectedValueOnce(new ApiError(code, 409, code))
    renderSession()
    fireEvent.click(questionAction("Prompt 1", "Delete turn"))
    confirm("Delete")
    await waitFor(() => expect(session.notify).toHaveBeenCalledWith(sentence, "error"))
    expect(screen.getByText("Prompt 1")).toBeInTheDocument()
  })

  it("shows any other refusal's reason, and an edit refused sends nothing", async () => {
    vi.mocked(deletePiTurn).mockRejectedValueOnce(new ApiError("off_branch", 400, "off_branch"))
    renderSession()
    edit("Prompt 1", "Prompt one, again")
    await waitFor(() => expect(session.notify).toHaveBeenCalledWith("off_branch", "error"))
    expect(session.send).not.toHaveBeenCalled()
  })

  it("offers neither edit nor delete while a run is live", () => {
    useTranscript(history(2), { running: true })
    renderSession()
    fireEvent.click(screen.getByRole("button", { name: "Prompt 0" }))
    for (const group of screen.queryAllByRole("group", { name: "Question actions" })) {
      expect(within(group).queryByRole("button", { name: "Edit message" })).toBeNull()
      expect(within(group).queryByRole("button", { name: "Delete turn" })).toBeNull()
    }
  })

  it("offers neither on a turn whose prompt has no entry yet", () => {
    const messages = history(2)
    useTranscript(messages.slice(0, 2), { unconfirmed: messages.slice(2) })
    renderSession()
    const groups = screen.getAllByRole("group", { name: "Question actions" })
    const pending = groups.find((group) => group.closest("[data-turn-key]")?.textContent?.includes("Prompt 1"))!
    expect(within(pending).queryByRole("button", { name: "Edit message" })).toBeNull()
    expect(within(pending).queryByRole("button", { name: "Delete turn" })).toBeNull()
  })
})
