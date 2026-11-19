import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { PiSessionEvent } from "../sessionStore"
import { startPiNotifications, type PiNotificationTabs } from "../notifications"

const store = vi.hoisted(() => ({
  listeners: new Set<(event: PiSessionEvent) => void>(),
  markPiSessionRead: vi.fn(),
}))

vi.mock("../sessionStore", () => ({
  markPiSessionRead: store.markPiSessionRead,
  piSessionEvents: {
    subscribe(listener: (event: PiSessionEvent) => void) {
      store.listeners.add(listener)
      return () => store.listeners.delete(listener)
    },
  },
}))

/** The Notification API as far as the notifications use it. */
class FakeNotification {
  static permission: NotificationPermission = "granted"
  static requestPermission = vi.fn(async () => FakeNotification.permission)
  static shown: FakeNotification[] = []
  onclick: (() => void) | null = null
  onclose: (() => void) | null = null
  closed = false

  constructor(readonly title: string, readonly options: NotificationOptions) {
    FakeNotification.shown.push(this)
  }

  close() {
    this.closed = true
    this.onclose?.()
  }
}

let visibility: DocumentVisibilityState = "visible"
let active: string | null = null
const tabs: PiNotificationTabs = {
  activeSession: () => active,
  titleOf: (sessionId) => sessionId === "s1" ? "Fix the parser" : null,
  open: vi.fn(),
}
let stop: () => void = () => {}

/** Emit `event` and let the permission check (a few promise hops) settle. */
async function emit(event: PiSessionEvent) {
  for (const listener of store.listeners) listener(event)
  for (let hop = 0; hop < 5; hop++) await Promise.resolve()
}

const shown = () => FakeNotification.shown.map(({ title, options, closed }) => ({ title, body: options.body, tag: options.tag, closed }))

beforeEach(() => {
  FakeNotification.permission = "granted"
  FakeNotification.shown = []
  FakeNotification.requestPermission.mockClear()
  store.markPiSessionRead.mockClear()
  vi.mocked(tabs.open).mockClear()
  visibility = "visible"
  active = null
  vi.stubGlobal("Notification", FakeNotification)
  vi.spyOn(document, "visibilityState", "get").mockImplementation(() => visibility)
  stop = startPiNotifications(tabs)
})

afterEach(() => {
  stop()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe("startPiNotifications", () => {
  it("notifies a finished run, a waiting dialog and a failed run, tagged by session and type", async () => {
    await emit({ type: "settled", sessionId: "s1", aborted: false })
    await emit({ type: "waiting", sessionId: "s1" })
    await emit({ type: "failed", sessionId: "s2" })
    expect(shown()).toEqual([
      { title: "Fix the parser", body: "Finished", tag: "s1:settled", closed: false },
      { title: "Fix the parser", body: "Waiting for your answer", tag: "s1:waiting", closed: false },
      { title: "Pi", body: "Run failed", tag: "s2:failed", closed: false },
    ])
  })

  it("stays silent for aborted runs and the settle of a failed run", async () => {
    await emit({ type: "settled", sessionId: "s1", aborted: true })
    await emit({ type: "failed", sessionId: "s1" })
    await emit({ type: "settled", sessionId: "s1", aborted: false })
    expect(shown().map(({ tag }) => tag)).toEqual(["s1:failed"])
    // The next run finishes normally
    await emit({ type: "settled", sessionId: "s1", aborted: false })
    expect(shown().map(({ tag }) => tag)).toEqual(["s1:failed", "s1:settled"])
  })

  it("replaces a repeat of the same session and type", async () => {
    await emit({ type: "settled", sessionId: "s1" })
    await emit({ type: "settled", sessionId: "s1" })
    expect(shown().map(({ tag, closed }) => [tag, closed])).toEqual([["s1:settled", true], ["s1:settled", false]])
  })

  it("skips the session in view and marks it read, but notifies once the page is hidden", async () => {
    active = "s1"
    await emit({ type: "settled", sessionId: "s1" })
    await emit({ type: "waiting", sessionId: "s1" })
    expect(shown()).toEqual([])
    expect(store.markPiSessionRead).toHaveBeenCalledWith("s1")
    await emit({ type: "waiting", sessionId: "s2" })
    expect(shown().map(({ tag }) => tag)).toEqual(["s2:waiting"])
    visibility = "hidden"
    await emit({ type: "settled", sessionId: "s1" })
    expect(shown().map(({ tag }) => tag)).toEqual(["s2:waiting", "s1:settled"])
  })

  it("marks the session in view read when the page becomes visible", () => {
    active = "s1"
    document.dispatchEvent(new Event("visibilitychange"))
    expect(store.markPiSessionRead).toHaveBeenCalledWith("s1")
  })

  it("focuses the window and the session's tab on click", async () => {
    const focus = vi.spyOn(window, "focus").mockImplementation(() => {})
    await emit({ type: "settled", sessionId: "s1" })
    FakeNotification.shown[0].onclick?.()
    expect(focus).toHaveBeenCalled()
    expect(tabs.open).toHaveBeenCalledWith("s1")
    expect(FakeNotification.shown[0].closed).toBe(true)
  })

  it("asks for permission on the first event and shows nothing when denied", async () => {
    FakeNotification.permission = "default"
    FakeNotification.requestPermission.mockImplementationOnce(async () => (FakeNotification.permission = "denied"))
    await emit({ type: "settled", sessionId: "s1" })
    expect(FakeNotification.requestPermission).toHaveBeenCalledTimes(1)
    expect(shown()).toEqual([])
    await emit({ type: "settled", sessionId: "s1" })
    expect(FakeNotification.requestPermission).toHaveBeenCalledTimes(1)
  })
})
