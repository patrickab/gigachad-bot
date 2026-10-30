import { afterEach, beforeEach, expect, it, vi } from "vitest"
import { LEARN_EMBED_CHANNEL, LEARN_PROTOCOL_VERSION, type LearnFrameMessage, type LearnShellMessage } from "@gigachad/chat-ui/learn-protocol"
import type { LearnView } from "@/lib/domain/types"
import { createLearnBridge, NAVIGATION_ACK_MS, parseFrameMessage } from "../bridge"

const VERSION = LEARN_PROTOCOL_VERSION
const hello = (theme: "light" | "dark") => ({ type: "hello", theme, version: VERSION })

function setup(theme: "light" | "dark" = "light") {
  const sent: LearnShellMessage[] = []
  const navigated: LearnView[] = []
  const onReady = vi.fn()
  const bridge = createLearnBridge(theme)
  bridge.connect({
    post: ({ channel, ...message }) => {
      expect(channel).toBe(LEARN_EMBED_CHANNEL)
      sent.push(message as LearnShellMessage)
    },
    onReady,
    onNavigate: (view) => navigated.push(view),
  })
  const frame = (message: LearnFrameMessage) => bridge.receive({ channel: LEARN_EMBED_CHANNEL, ...message })
  return { bridge, sent, navigated, onReady, frame }
}

beforeEach(() => vi.useFakeTimers())
afterEach(() => vi.useRealTimers())

it("greets every ready with the current theme and its version, then shows the wanted view", () => {
  const { bridge, sent, onReady, frame } = setup("light")
  bridge.show({ kind: "session", sessionId: "s1" })
  bridge.setTheme("dark")
  expect(sent).toEqual([])

  frame({ type: "ready", version: VERSION })
  expect(onReady).toHaveBeenCalledOnce()
  expect(sent).toEqual([hello("dark"), { type: "open-session", sessionId: "s1" }])

  frame({ type: "ready", version: VERSION })
  expect(sent.slice(2)).toEqual([hello("dark"), { type: "open-session", sessionId: "s1" }])
})

it("switches the theme live once the frame is ready, only when it changes", () => {
  const { bridge, sent, frame } = setup("light")
  frame({ type: "ready", version: VERSION })
  sent.length = 0

  bridge.setTheme("dark")
  bridge.setTheme("dark")
  bridge.setTheme("light")
  expect(sent).toEqual([{ type: "set-theme", theme: "dark" }, { type: "set-theme", theme: "light" }])
})

it("switches views by message and skips a view the frame already shows", () => {
  const { bridge, sent, frame } = setup()
  bridge.show({ kind: "home" })
  frame({ type: "ready", version: VERSION })
  frame({ type: "view", sessionId: null })
  sent.length = 0

  bridge.show({ kind: "home" })
  bridge.show({ kind: "new" })
  frame({ type: "view", sessionId: null })
  bridge.show({ kind: "session", sessionId: "s1" })
  frame({ type: "view", sessionId: "s1" })
  bridge.show({ kind: "session", sessionId: "s1" })
  bridge.show({ kind: "home" })
  expect(sent).toEqual([{ type: "new-session", kind: "teach" }, { type: "open-session", sessionId: "s1" }, { type: "open-home" }])
})

it("starts a practice session by its kind and tells it apart from a new teaching session", () => {
  const { bridge, sent, frame } = setup()
  bridge.show({ kind: "new" })
  frame({ type: "ready", version: VERSION })
  frame({ type: "view", sessionId: null })
  sent.length = 0

  bridge.show({ kind: "new", sessionKind: "practice" })
  frame({ type: "view", sessionId: null })
  bridge.show({ kind: "new", sessionKind: "practice" })
  bridge.show({ kind: "new", sessionKind: "teach" })
  expect(sent).toEqual([{ type: "new-session", kind: "practice" }, { type: "new-session", kind: "teach" }])
})

it("reports the session the user opened inside the frame, and home when they leave a session", () => {
  const { bridge, navigated, frame } = setup()
  bridge.show({ kind: "home" })
  frame({ type: "ready", version: VERSION })
  frame({ type: "view", sessionId: null })

  frame({ type: "view", sessionId: "picked" })
  expect(navigated).toEqual([{ kind: "session", sessionId: "picked" }])

  bridge.show({ kind: "session", sessionId: "picked" })
  frame({ type: "view", sessionId: null })
  expect(navigated.at(-1)).toEqual({ kind: "home" })
})

it("adopts the session a new-session view turns into", () => {
  const { bridge, navigated, frame } = setup()
  bridge.show({ kind: "new" })
  frame({ type: "ready", version: VERSION })
  frame({ type: "view", sessionId: null })
  vi.advanceTimersByTime(NAVIGATION_ACK_MS)

  frame({ type: "view", sessionId: "created" })
  expect(navigated).toEqual([{ kind: "session", sessionId: "created" }])
})

it("ignores stale reports while the frame catches up with a navigation", () => {
  const { bridge, navigated, frame } = setup()
  bridge.show({ kind: "session", sessionId: "s1" })
  frame({ type: "ready", version: VERSION })
  // The reload's own report of `/` arrives before the frame followed `open-session`.
  frame({ type: "view", sessionId: null })
  expect(navigated).toEqual([])

  vi.advanceTimersByTime(NAVIGATION_ACK_MS)
  frame({ type: "view", sessionId: "other" })
  expect(navigated).toEqual([{ kind: "session", sessionId: "other" }])
})

it("keeps reports to itself while no learn tab shows the frame", () => {
  const { bridge, navigated, sent, frame } = setup()
  frame({ type: "ready", version: VERSION })
  frame({ type: "view", sessionId: "s1" })
  expect(navigated).toEqual([])
  expect(sent).toEqual([hello("light")])

  bridge.show({ kind: "session", sessionId: "s1" })
  expect(sent).toHaveLength(1)
})

it("ignores messages off the learn channel and waits for ready after a reset", () => {
  const { bridge, sent, onReady, frame } = setup()
  bridge.receive({ type: "ready", version: VERSION })
  bridge.receive({ channel: "omp-embed", type: "ready", version: VERSION })
  bridge.receive(null)
  expect(onReady).not.toHaveBeenCalled()

  frame({ type: "ready", version: VERSION })
  bridge.reset()
  sent.length = 0
  bridge.setTheme("dark")
  bridge.show({ kind: "home" })
  expect(sent).toEqual([])
})

it("drives the frame only at the same protocol version, and keeps theme and navigation at any version", () => {
  const { bridge, sent, onReady, frame } = setup()
  const closeModal = { type: "close-modal" } as const
  expect(bridge.drive()).toBe("waiting")
  expect(bridge.send(closeModal)).toBe(false)

  // A frame of the first protocol sends no version: theme and navigation still work, drive messages stay home.
  bridge.receive({ channel: LEARN_EMBED_CHANNEL, type: "ready" })
  expect(onReady).toHaveBeenCalledOnce()
  expect(bridge.drive()).toBe("mismatch")
  bridge.show({ kind: "new" })
  expect(bridge.send(closeModal)).toBe(false)
  expect(sent).toEqual([hello("light"), { type: "new-session", kind: "teach" }])

  frame({ type: "ready", version: VERSION + 1 })
  expect(bridge.drive()).toBe("mismatch")

  frame({ type: "ready", version: VERSION })
  expect(bridge.drive()).toBe("matched")
  sent.length = 0
  expect(bridge.send(closeModal)).toBe(true)
  expect(bridge.send({ type: "set-thinking", sessionId: null, level: "high" })).toBe(true)
  expect(sent).toEqual([closeModal, { type: "set-thinking", sessionId: null, level: "high" }])

  bridge.reset()
  expect(bridge.drive()).toBe("waiting")
  expect(bridge.send(closeModal)).toBe(false)
})

it("keeps the latest state per type, session meta per session, until the frame reloads", () => {
  const { bridge, frame } = setup()
  const changes = vi.fn()
  bridge.subscribe(changes)
  frame({ type: "ready", version: VERSION })
  expect(bridge.state("model")).toBeNull()
  expect(bridge.state("session-meta")).toEqual({})

  const model = { type: "model" as const, sessionId: "s1", model: null, thinkingLevel: null, thinkingLevels: [], locked: true, unavailable: false }
  frame(model)
  frame({ ...model, unavailable: true })
  expect(bridge.state("model")).toEqual({ ...model, unavailable: true })
  const kept = bridge.state("model")
  frame({ type: "modal", open: true })
  expect(bridge.state("model")).toBe(kept)

  const meta = (sessionId: string, title: string) =>
    ({ type: "session-meta", sessionId, title, kind: "practice", phase: null, status: null, updatedAt: "2026-10-29T10:00:00.000Z" }) as const
  frame(meta("s1", "Limits"))
  frame(meta("s2", "Series"))
  frame(meta("s1", "Limits, part 2"))
  expect(bridge.state("session-meta")).toEqual({ s1: meta("s1", "Limits, part 2"), s2: meta("s2", "Series") })
  frame({ type: "view", sessionId: "s1" })
  expect(bridge.state("view")).toEqual({ type: "view", sessionId: "s1" })
  expect(changes.mock.calls.length).toBeGreaterThanOrEqual(7)

  frame({ type: "ready", version: VERSION })
  expect(bridge.state("model")).toBeNull()
  expect(bridge.state("session-meta")).toEqual({})
  frame({ type: "modal", open: false })
  bridge.reset()
  expect(bridge.state("modal")).toBeNull()
})

it("takes only the view from a frame at another version: its other messages may mean something else", () => {
  const { bridge, frame, navigated } = setup()
  const keys = vi.fn()
  bridge.onEvent("chrome-key", keys)
  bridge.show({ kind: "session", sessionId: "s1" })
  frame({ type: "ready", version: VERSION + 1 })
  vi.advanceTimersByTime(NAVIGATION_ACK_MS + 1)

  frame({ type: "session-meta", sessionId: "s1", title: "Renamed", kind: "teach", phase: "probe", status: null, updatedAt: "2026-10-29T10:00:00.000Z" })
  frame({ type: "activity", items: [] })
  frame({ type: "chrome-key", action: "toggle-sidebar" })
  frame({ type: "view", sessionId: "s2" })
  expect(bridge.state("session-meta")).toEqual({})
  expect(bridge.state("activity")).toBeNull()
  expect(keys).not.toHaveBeenCalled()
  expect(bridge.state("view")).toEqual({ type: "view", sessionId: "s2" })
  expect(navigated).toEqual([{ kind: "session", sessionId: "s2" }])

  frame({ type: "ready", version: VERSION })
  frame({ type: "chrome-key", action: "toggle-sidebar" })
  expect(keys).toHaveBeenCalledWith({ type: "chrome-key", action: "toggle-sidebar" })
})

it("hands frame events to their listeners only, never keeping them", () => {
  const { bridge, frame } = setup()
  const opens = vi.fn()
  const stop = bridge.onEvent("model-open", opens)
  frame({ type: "ready", version: VERSION })

  frame({ type: "model-open", sessionId: "s1" })
  expect(opens).toHaveBeenCalledWith({ type: "model-open", sessionId: "s1" })

  stop()
  frame({ type: "model-open", sessionId: null })
  expect(opens).toHaveBeenCalledOnce()
})

it("drops malformed frame messages and strips fields the protocol does not name", () => {
  const on = (message: object) => parseFrameMessage({ channel: LEARN_EMBED_CHANNEL, ...message })
  const ref = { provider: "custom", modelId: "gemini/gemini-3.5-flash" }
  const model = { type: "model", sessionId: null, model: ref, thinkingLevel: "high", thinkingLevels: ["low", "high"], locked: false, unavailable: false }
  const activity = { type: "activity", items: [{ id: "a1", label: "Researcher", status: "running", sessionId: null }] }

  expect(on({ type: "ready", version: 2, extra: true })).toEqual({ type: "ready", version: 2 })
  expect(on({ type: "ready" })).toEqual({ type: "ready", version: 1 })
  expect(on(model)).toEqual(model)
  expect(on({ ...model, model: null, thinkingLevel: null })).toEqual({ ...model, model: null, thinkingLevel: null })
  expect(on({ type: "models", served: [ref, { ...ref, extra: 1 }] })).toEqual({ type: "models", served: [ref, ref] })
  expect(on(activity)).toEqual(activity)
  const meta = { type: "session-meta", sessionId: "s1", title: "", kind: null, phase: "probe", status: "waiting", updatedAt: "2026-10-29T10:00:00.000Z" }
  expect(on(meta)).toEqual(meta)
  expect(on({ ...meta, status: null })).toEqual({ ...meta, status: null })
  expect(on({ type: "session-plan", sessionId: "s1", available: true, extra: 1 })).toEqual({ type: "session-plan", sessionId: "s1", available: true })
  expect(on({ type: "chrome-key", action: "rename-session" })).toEqual({ type: "chrome-key", action: "rename-session" })

  for (const malformed of [
    { type: "ready", version: 0 },
    { type: "ready", version: "2" },
    { type: "view", sessionId: "" },
    { type: "view" },
    { ...model, model: { provider: "custom" } },
    { ...model, thinkingLevels: "high" },
    { ...model, thinkingLevels: ["high", 3] },
    { ...model, locked: undefined },
    { type: "models", served: [ref, null] },
    { type: "models" },
    { type: "model-open", sessionId: 7 },
    { type: "modal", open: 1 },
    { ...meta, kind: "lecture" },
    { ...meta, sessionId: null },
    { ...meta, status: "idle" },
    { ...meta, status: undefined },
    { ...meta, updatedAt: "yesterday" },
    { type: "session-plan", sessionId: "s1", available: "yes" },
    { type: "session-plan", sessionId: "", available: true },
    { type: "chrome-key", action: "jump-graph" },
    { type: "chrome-key" },
    { type: "activity", items: [{ id: "a1", label: "x", status: "done" }] },
    { type: "activity", items: null },
    { type: "unknown" },
    { type: "toString" },
  ]) {
    expect(on(malformed), JSON.stringify(malformed)).toBeNull()
  }
  expect(parseFrameMessage({ channel: "omp-embed", type: "modal", open: true })).toBeNull()
  expect(parseFrameMessage("learn-embed")).toBeNull()
})
