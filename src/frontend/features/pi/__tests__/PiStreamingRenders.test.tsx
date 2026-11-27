import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { Profiler, useReducer, type ComponentProps } from "react"
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest"
import type { PiSessionInfo } from "@/lib/backend/api"
import { PiTab } from "../PiTab"
import type { SessionSubscriber } from "../piSocket"
import { PiTranscript } from "../PiTranscript"
import { EMPTY_TRANSCRIPT, transcriptReducer, type Transcript, type TranscriptAction } from "../transcript"
import type { AgentMessage } from "../types"

/**
 * A streamed delta re-renders the live turn only, wired as `PiTab` wires the transcript: the session's reducer, stable
 * callbacks and file links. Text is counted where it renders (`LaTeXMarkdown`), for an open settled turn and a collapsed one.
 */

const { renders, durations, socket, stable } = vi.hoisted(() => ({
  renders: new Map<string, number>(),
  /** Summed `actualDuration` per `<Profiler>` id, while `durations.measuring` */
  durations: { measuring: false, ms: new Map<string, number>() },
  /** The session's subscriber on the page's pi-host socket */
  socket: { subscriber: null as SessionSubscriber | null },
  // Stable identities, like the real providers'
  stable: {
    enabledTools: { enabledTools: [], toggleTool: () => {} },
    settings: { ocrModel: "" },
    neovim: { available: false, enabled: false, setPreferred: () => {} },
    status: { running: true, waiting: false, unread: false, failed: false },
  },
}))
const count = vi.hoisted(() => (key: string) => { renders.set(key, (renders.get(key) ?? 0) + 1) })
const profile = vi.hoisted(() => (id: string, _phase: string, actualDuration: number) => {
  if (durations.measuring) durations.ms.set(id, (durations.ms.get(id) ?? 0) + actualDuration)
})
vi.mock("@/components/content/LaTeXMarkdown", () => ({
  LaTeXMarkdown: ({ content }: { content: string }) => {
    count(content)
    return <div>{content}</div>
  },
}))

// A Pi tab with a live session: the network, pi-host's socket and the app-wide providers stubbed
vi.mock("../piSocket", async (importOriginal) => ({
  ...await importOriginal<typeof import("../piSocket")>(),
  subscribePiSession: (_sessionId: string, subscriber: SessionSubscriber) => {
    socket.subscriber = subscriber
    return () => { socket.subscriber = null }
  },
  // pi-host answers what the session asks on attach: its model, the models and levels it offers, empty commands and stats
  sendPiCommand: (sessionId: string, command: { id: string; type: string }) => {
    const model = { provider: "anthropic", id: "claude" }
    const data: Record<string, unknown> = {
      get_commands: { commands: [] },
      get_available_models: { models: [model] },
      get_available_thinking_levels: { levels: ["off", "medium", "high"] },
      get_session_stats: { cost: 0 },
      get_state: { model, thinkingLevel: "medium", isStreaming: true, sessionId },
    }
    queueMicrotask(() => socket.subscriber?.onFrame({ type: "pi", sessionId, record: { type: "response", id: command.id, success: command.type in data, data: data[command.type] } }))
    return true
  },
  sendPiFrame: () => true,
  observePiSocket: () => () => {},
}))
vi.mock("@/lib/backend/api", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/lib/backend/api")>(),
  fetchPiSessions: async () => [session],
  openPiSession: async () => session,
  fetchPiFiles: async () => [],
  callFactory: async () => ({ waves: [] }),
}))
vi.mock("@/hooks/useResource", () => ({
  useResource: () => undefined,
  useResourceState: () => ({ value: undefined, error: null }),
  useResourceFamily: () => ({}),
}))
vi.mock("../sessionStore", async (importOriginal) => ({
  ...await importOriginal<typeof import("../sessionStore")>(),
  refreshPiSessions: () => {},
  usePiSessionStatus: () => stable.status,
}))
vi.mock("@/features/chat/EnabledToolsContext", () => ({ useEnabledTools: () => stable.enabledTools }))
vi.mock("@/contexts/SettingsContext", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/contexts/SettingsContext")>(),
  useSettings: () => stable.settings,
  useNeovimSetting: () => stable.neovim,
}))
vi.mock("@/features/editor/FilePanel", () => ({ useOpenFile: () => () => {} }))

// The top bar and the composer, counted and profiled around the real components. Pi's top bar renders into the bar's slots through portals
vi.mock("../PiTopBar", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../PiTopBar")>()
  const { Profiler } = await import("react")
  return { ...actual, PiTopBar: (props: ComponentProps<typeof actual.PiTopBar>) => <Profiler id="bar" onRender={profile}><actual.PiTopBar {...props} /></Profiler> }
})
vi.mock("@/features/settings/ModelDropdown", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/features/settings/ModelDropdown")>()
  return { ...actual, ModelDropdown: (props: ComponentProps<typeof actual.ModelDropdown>) => { count("ModelDropdown"); return <actual.ModelDropdown {...props} /> } }
})
vi.mock("../PiComposer", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../PiComposer")>()
  const { Profiler } = await import("react")
  return {
    ...actual,
    PiComposer: (props: ComponentProps<typeof actual.PiComposer>) => {
      count("PiComposer")
      return <Profiler id="composer" onRender={profile}><actual.PiComposer {...props} /></Profiler>
    },
  }
})

beforeAll(() => { Element.prototype.scrollIntoView = () => {} })
vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} })
// The composer keeps its draft there, jsdom under this Node has none
const stored = new Map<string, string>()
vi.stubGlobal("localStorage", {
  getItem: (key: string) => stored.get(key) ?? null,
  setItem: (key: string, value: string) => { stored.set(key, value) },
  removeItem: (key: string) => { stored.delete(key) },
})

const settled: AgentMessage[] = [0, 1, 2].flatMap((index): AgentMessage[] => [
  { role: "user", content: `Prompt ${index}`, timestamp: index * 10 },
  { role: "assistant", content: [{ type: "text", text: `Answer ${index}` }], provider: "anthropic", model: "claude", stopReason: "stop", timestamp: index * 10 + 1 },
])
const initial = { ...EMPTY_TRANSCRIPT, messages: settled, entryIds: settled.map((_, index) => `e${index}`) }

const fileLinks = { base: "/repo", open: () => {} }
const noop = () => {}

/** The session's reducer, and `set` for a state the host's resync brings (an entry confirmed, a turn deleted). */
type Action = TranscriptAction | { kind: "set"; update: (transcript: Transcript) => Transcript }
const reducer = (transcript: Transcript, action: Action) => (action.kind === "set" ? action.update(transcript) : transcriptReducer(transcript, action))

function Session({ onDispatch }: { onDispatch: (dispatch: (action: Action) => void) => void }) {
  const [transcript, dispatch] = useReducer(reducer, initial)
  onDispatch(dispatch)
  return <PiTranscript transcript={transcript} held={[]} onRetryHeld={noop} onDeleteHeld={noop} fileLinks={fileLinks} onBranch={noop} onDeleteTurn={noop} onEditTurn={noop} onColumnChange={noop} />
}

const event = (record: Record<string, unknown>): TranscriptAction => ({ kind: "event", event: { type: String(record.type), ...record } })
const delta = (text: string) => event({ type: "message_update", assistantMessageEvent: { type: "text_delta", contentIndex: 0, delta: text } })

describe("a Pi session while its answer streams", () => {
  afterEach(cleanup)

  it("re-renders neither an open nor a collapsed settled turn", () => {
    let dispatch!: (action: Action) => void
    render(<Session onDispatch={(current) => { dispatch = current }} />)
    // Prompt 0 open, Prompt 1 collapsed to its answer (rendered at once: jsdom has no IntersectionObserver)
    fireEvent.click(screen.getByRole("button", { name: "Prompt 0" }))

    act(() => dispatch({ kind: "prompt", prompt: { role: "user", content: "Prompt 3", timestamp: 100 } }))
    act(() => dispatch(event({ type: "agent_start" })))
    act(() => dispatch(event({ type: "message_start", message: { role: "assistant", content: [], provider: "anthropic", model: "claude", stopReason: "pending", timestamp: 101 } })))
    act(() => dispatch(event({ type: "message_update", assistantMessageEvent: { type: "text_start", contentIndex: 0 } })))
    act(() => dispatch(delta("A")))
    expect(renders.get("A")).toBeGreaterThan(0)

    const watched = ["Prompt 0", "Answer 0", "Answer 1"]
    const before = watched.map((key) => renders.get(key) ?? 0)
    expect(before.every((count) => count > 0)).toBe(true)

    let answer = "A"
    for (const token of ["n", "s", "w", "e", "r"]) {
      answer += token
      act(() => dispatch(delta(token)))
      expect(renders.get(answer)).toBeGreaterThan(0)
    }
    expect(watched.map((key) => renders.get(key) ?? 0)).toEqual(before)
  })

  it("re-renders no settled turn for a send from start to end, nor for the delete of another turn", () => {
    let dispatch!: (action: Action) => void
    render(<Session onDispatch={(current) => { dispatch = current }} />)
    fireEvent.click(screen.getByRole("button", { name: "Prompt 1" }))
    const watched = ["Prompt 1", "Answer 1", "Answer 0"]
    const before = watched.map((key) => renders.get(key) ?? 0)
    expect(before.every((count) => count > 0)).toBe(true)

    const prompt: AgentMessage = { role: "user", content: "Prompt 3", timestamp: 100 }
    const reply: AgentMessage = { role: "assistant", content: [{ type: "text", text: "Reply" }], provider: "anthropic", model: "claude", stopReason: "stop", timestamp: 101 }
    act(() => dispatch({ kind: "prompt", prompt: { role: "user", content: "Prompt 3", timestamp: 100 } }))
    act(() => dispatch(event({ type: "agent_start" })))
    act(() => dispatch(event({ type: "message_end", message: prompt })))
    act(() => dispatch(event({ type: "message_start", message: { ...reply, content: [], stopReason: "pending" } })))
    act(() => dispatch(event({ type: "message_update", assistantMessageEvent: { type: "text_start", contentIndex: 0 } })))
    act(() => dispatch(delta("Reply")))
    act(() => dispatch(event({ type: "message_end", message: reply })))
    act(() => dispatch(event({ type: "tool_execution_start", toolCallId: "elsewhere" })))
    act(() => dispatch(event({ type: "agent_settled" })))
    // The host's resync confirms both entries
    act(() => dispatch({ kind: "set", update: (t) => ({ ...t, messages: [...t.messages, ...t.unconfirmed], entryIds: [...t.entryIds, "e6", "e7"], unconfirmed: [] }) }))
    expect(renders.get("Reply")).toBeGreaterThan(0)
    expect(watched.map((key) => renders.get(key) ?? 0)).toEqual(before)

    // An earlier turn and a later one go, as the host's resync after a delete leaves the transcript
    act(() => dispatch({ kind: "set", update: (t) => ({ ...t, messages: t.messages.slice(2), entryIds: t.entryIds.slice(2) }) }))
    act(() => dispatch({ kind: "set", update: (t) => ({ ...t, messages: t.messages.slice(0, -2), entryIds: t.entryIds.slice(0, -2) }) }))
    expect(screen.queryByText("Prompt 0")).toBeNull()
    expect(screen.queryByText("Prompt 3")).toBeNull()
    expect(watched.map((key) => renders.get(key) ?? 0)).toEqual(before)
  })
})

/** A settled turn of 4 messages: prompt, a bash call, its result, the answer. */
function toolTurn(index: number): AgentMessage[] {
  const id = `call-${index}`
  return [
    { role: "user", content: `Task ${index}`, timestamp: index * 10 },
    { role: "assistant", content: [{ type: "toolCall", id, name: "bash", arguments: { command: "ls" } }], provider: "anthropic", model: "claude", stopReason: "toolUse", timestamp: index * 10 + 1 },
    { role: "toolResult", toolCallId: id, toolName: "bash", content: [{ type: "text", text: "out" }], isError: false, timestamp: index * 10 + 2 },
    { role: "assistant", content: [{ type: "text", text: `Done ${index}` }], provider: "anthropic", model: "claude", stopReason: "stop", timestamp: index * 10 + 3 },
  ]
}

/**
 * Index reads of the session's message list per `tool_execution_update` frame: `settledTurns` turns of 4 messages and a live
 * turn whose bash call streams output. The list is a counting `Proxy`; `unconfirmed` and `optimistic` stay empty, else
 * `shownMessages` copies it and the reads go to the copy.
 */
function messageReadsPerToolOutputFrame(settledTurns: number, frames: number): number {
  const live: AgentMessage[] = [
    { role: "user", content: "Live task", timestamp: 100_000 },
    { role: "assistant", content: [{ type: "toolCall", id: "call-live", name: "bash", arguments: { command: "make" } }], provider: "anthropic", model: "claude", stopReason: "toolUse", timestamp: 100_001 },
  ]
  const list = [...Array.from({ length: settledTurns }, (_, index) => toolTurn(index)).flat(), ...live]
  let reads = 0
  const messages = new Proxy(list, {
    get(target, key, receiver) {
      if (typeof key === "string" && /^\d+$/.test(key)) reads++
      return Reflect.get(target, key, receiver)
    },
  })
  const start = transcriptReducer(
    { ...EMPTY_TRANSCRIPT, messages, entryIds: list.map((_, index) => `e${index}`), running: true },
    event({ type: "tool_execution_start", toolCallId: "call-live" }),
  )
  let dispatch!: (action: TranscriptAction) => void
  function Live({ onDispatch }: { onDispatch: (dispatch: (action: TranscriptAction) => void) => void }) {
    const [transcript, next] = useReducer(transcriptReducer, start)
    onDispatch(next)
    return <PiTranscript transcript={transcript} fileLinks={fileLinks} onBranch={noop} onDeleteTurn={noop} onEditTurn={noop} />
  }
  render(<Live onDispatch={(current) => { dispatch = current }} />)
  reads = 0
  for (let frame = 0; frame < frames; frame++) {
    act(() => dispatch({ kind: "events", events: [{ type: "tool_execution_update", toolCallId: "call-live", partialResult: { content: [{ type: "text", text: `line ${frame}` }] } }] }))
  }
  expect(screen.getByText(`line ${frames - 1}`, { exact: false })).toBeTruthy()
  const perFrame = reads / frames
  // The end still reaches the live turn's row: done, with its duration
  act(() => dispatch({ kind: "events", events: [{ type: "tool_execution_end", toolCallId: "call-live", result: { content: [{ type: "text", text: "built" }] }, durationMs: 4200 }] }))
  expect(screen.getByText("4.2s")).toBeTruthy()
  cleanup()
  return perFrame
}

describe("a Pi session while a tool streams output", () => {
  afterEach(cleanup)

  it("reads the message list per tool-output frame (measure)", () => {
    // 50 settled turns of 4 messages + a live turn of 2: only the live turn's 2 messages are read (its `turnView`);
    // settled turns keep their views and committed runs are not re-derived. Was 404 (every message twice) before task pi-live.
    expect(messageReadsPerToolOutputFrame(50, 10)).toBe(2)
    expect(messageReadsPerToolOutputFrame(200, 10)).toBe(2)
  })
})

const session: PiSessionInfo = {
  sessionId: "s1", sessionPath: "/sessions/s1.jsonl", cwd: "/home/u/git/repo", machine: "local", createdAt: "", modifiedAt: "",
  firstMessage: "", messageCount: 0, tokens: 0, live: null, pinned: false, archived: false,
}

const modelProps = {
  models: { ollama: [], providers: [], defaults: { default_model: "", small_model: "", vision_model: "", sandbox_model: "" }, tab_order: [], unavailable: [] },
  onProvidersChange: async () => {}, onDefaultsChange: async () => {}, placement: "below" as const,
}

describe("a Pi tab while its answer streams", () => {
  afterEach(() => {
    cleanup()
    renders.clear()
  })

  it("renders the top bar and the composer per delta frame (measure)", async () => {
    // A session of 20 settled turns, attached in a Pi tab with its top bar slots
    const messages = Array.from({ length: 20 }, (_, index): AgentMessage[] => [
      { role: "user", content: `Prompt ${index}`, timestamp: index * 10 },
      { role: "assistant", content: [{ type: "text", text: `Answer ${index}` }], provider: "anthropic", model: "claude", stopReason: "stop", timestamp: index * 10 + 1 },
    ]).flat()
    const entries = messages.map((message, index) => ({ type: "message", id: `e${index}`, parentId: index ? `e${index - 1}` : null, timestamp: new Date(message.timestamp).toISOString(), message }))
    const topBar = document.body.appendChild(document.createElement("div"))
    const slots = { thinking: topBar.appendChild(document.createElement("div")), model: topBar.appendChild(document.createElement("div")), usage: topBar.appendChild(document.createElement("div")) }
    render(
      <Profiler id="tab" onRender={profile}>
        <PiTab view={{ repo: "repo", sessionId: "s1" }} onViewChange={noop} modelProps={modelProps} slots={slots} />
      </Profiler>,
    )
    await waitFor(() => expect(socket.subscriber).not.toBeNull())
    const frame = (record: Record<string, unknown>) => act(() => socket.subscriber!.onFrame({ type: "pi", sessionId: "s1", record }))
    act(() => socket.subscriber!.onFrame({
      type: "attached", sessionId: "s1", streamId: "x", sequence: 0,
      state: { isStreaming: true, sessionId: "s1", model: { provider: "anthropic", id: "claude" }, thinkingLevel: "medium" },
      entries: { entries, leafId: entries.at(-1)!.id }, ui: { dialogs: [], statuses: {}, queue: { steering: [], followUp: [] } },
    }))
    await waitFor(() => expect(renders.get("Answer 19")).toBeGreaterThan(0))
    await waitFor(() => expect(renders.get("ModelDropdown")).toBeGreaterThan(0))
    frame({ type: "message_start", message: { role: "assistant", content: [], provider: "anthropic", model: "claude", stopReason: "pending", timestamp: 1000 } })
    frame({ type: "message_update", assistantMessageEvent: { type: "text_start", contentIndex: 0 } })

    // One delta per frame: the coalescer applies it on the next animation frame
    let answer = ""
    const delta = async () => {
      answer += "b"
      frame({ type: "message_update", assistantMessageEvent: { type: "text_delta", contentIndex: 0, delta: "b" } })
      await waitFor(() => expect(renders.get(answer)).toBeGreaterThan(0))
    }
    await delta()
    await delta()
    const watched = ["ModelDropdown", "PiComposer"] as const
    const before = Object.fromEntries(watched.map((key) => [key, renders.get(key) ?? 0]))
    durations.ms.clear()
    durations.measuring = true
    for (let index = 0; index < 100; index++) await delta()
    durations.measuring = false
    const counts = Object.fromEntries(watched.map((key) => [key, (renders.get(key) ?? 0) - before[key]]))
    const { bar = 0, composer = 0, tab: total = 0 } = Object.fromEntries(durations.ms)
    console.info(`pi, 100 delta frames: ${JSON.stringify(counts)}; bar ${(100 * bar / total).toFixed(1)}%, composer ${(100 * composer / total).toFixed(1)}% of ${total.toFixed(0)} ms`)

    // baseline, lowered by the live-session selectors
    expect(counts).toEqual({ ModelDropdown: 100, PiComposer: 100 })
    // WorkspaceSurface and the tab workspace sit above PiTab: measured 0 per frame in the full TabContent, nothing to lower
    topBar.remove()
  })
})
