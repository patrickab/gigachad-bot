/**
 * System notifications for Pi runs, started once by `AppShell` (port of omp-web's notification rules, MIT,
 * `~/git/ompweb_original/docs/specs/2026-10-06-notifications-design.md`).
 *
 * - In-page only (Notification API), no Web Push. Permission is asked on the first event that would notify
 * - "Finished" on `settled`, "Waiting for your answer" on `waiting`, "Run failed" on `failed`. Aborted runs and the
 *   settle of a failed run stay silent
 * - Nothing for the session the frontmost tab shows while the page is visible; that session is marked read instead
 * - Tag `<sessionId>:<type>`: a repeat replaces the previous notification. Clicking one focuses the window and the session's tab
 */
import { markPiSessionRead, piSessionEvents, type PiSessionEvent } from "./sessionStore"

/** What the notifications need from the tabs. */
export interface PiNotificationTabs {
  /** The session the frontmost tab shows, null when it shows none. */
  activeSession: () => string | null
  /** The title of the tab showing `sessionId`, null when no tab shows it. */
  titleOf: (sessionId: string) => string | null
  /** Bring the tab showing `sessionId` to the front. */
  open: (sessionId: string) => void
}

const BODIES: Record<PiSessionEvent["type"], string> = {
  settled: "Finished",
  waiting: "Waiting for your answer",
  failed: "Run failed",
}

/** Whether `sessionId` is in view: the page is visible and its frontmost tab shows it. */
function inView(tabs: PiNotificationTabs, sessionId: string): boolean {
  return document.visibilityState === "visible" && tabs.activeSession() === sessionId
}

let asking: Promise<NotificationPermission> | null = null

/** Whether notifications may show, asking once while the user has not decided. */
async function allowed(): Promise<boolean> {
  if (typeof Notification === "undefined") return false
  if (Notification.permission === "default") {
    asking ??= Notification.requestPermission().finally(() => { asking = null })
    await asking
  }
  return Notification.permission === "granted"
}

/** Notify about the run events of the sessions this page attached; returns the stop. */
export function startPiNotifications(tabs: PiNotificationTabs): () => void {
  const shown = new Map<string, Notification>()
  // Sessions whose current run failed: its settle is no "Finished"
  const failed = new Set<string>()

  const show = async (event: PiSessionEvent) => {
    if (!(await allowed()) || inView(tabs, event.sessionId)) return
    const tag = `${event.sessionId}:${event.type}`
    shown.get(tag)?.close()
    const notification = new Notification(tabs.titleOf(event.sessionId) ?? "Pi", { body: BODIES[event.type], tag })
    shown.set(tag, notification)
    notification.onclick = () => {
      window.focus()
      tabs.open(event.sessionId)
      notification.close()
    }
    notification.onclose = () => {
      if (shown.get(tag) === notification) shown.delete(tag)
    }
  }

  const stopEvents = piSessionEvents.subscribe((event) => {
    if (event.type === "failed") failed.add(event.sessionId)
    if (event.type === "settled") {
      const runFailed = failed.delete(event.sessionId)
      if (inView(tabs, event.sessionId)) markPiSessionRead(event.sessionId)
      if (runFailed || event.aborted) return
    }
    if (inView(tabs, event.sessionId)) return
    void show(event)
  })

  // Coming back to the page reads the session in view
  const readInView = () => {
    const sessionId = tabs.activeSession()
    if (sessionId && document.visibilityState === "visible") markPiSessionRead(sessionId)
  }
  document.addEventListener("visibilitychange", readInView)

  return () => {
    stopEvents()
    document.removeEventListener("visibilitychange", readInView)
    for (const notification of shown.values()) notification.close()
  }
}
