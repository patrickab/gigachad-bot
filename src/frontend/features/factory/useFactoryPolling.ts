import { useEffect, useEffectEvent, useRef, useState } from "react"
import { useTabActive } from "@/contexts/TabContext"
import { fetchFactoryEvents, fetchLatestFactoryEventSeq } from "@/lib/backend/api"
import { errorMessage } from "@/lib/utils"

export const POLL_INTERVAL_MS = 3000

/**
 * Keep `refresh` in step with the engine while the tab is frontmost and the page visible.
 *
 * - Runs `refresh(null)` once the event cursor is set (null = reload everything)
 * - Polls `/factory/events` every 3 seconds, then `refresh(ids)` with the tasks that had events
 * - Returns the last event-polling error, cleared by the next successful poll
 */
export function useFactoryPolling(repo: string, refresh: (changedTaskIds: ReadonlySet<string> | null) => Promise<void> | void): string | null {
  const active = useTabActive()
  const [visible, setVisible] = useState(() => document.visibilityState === "visible")
  const [error, setError] = useState<string | null>(null)
  const cursor = useRef<number | null>(null)
  const onRefresh = useEffectEvent(refresh)

  useEffect(() => {
    const update = () => setVisible(document.visibilityState === "visible")
    document.addEventListener("visibilitychange", update)
    return () => document.removeEventListener("visibilitychange", update)
  }, [])

  const polling = active && visible
  useEffect(() => {
    if (!polling) return
    let cancelled = false
    let busy = false

    const tick = async () => {
      if (busy) return
      busy = true
      try {
        // The cursor only advances once `refresh` succeeded, so a failed refresh is retried by the next tick.
        if (cursor.current === null) {
          const latest = await fetchLatestFactoryEventSeq()
          if (cancelled) return
          await onRefresh(null)
          cursor.current = latest
        } else {
          const events = await fetchFactoryEvents(repo, cursor.current)
          if (cancelled) return
          if (events.length > 0) {
            await onRefresh(new Set(events.map((event) => event.task_id)))
            cursor.current = events[events.length - 1].seq
          }
        }
        if (!cancelled) setError(null)
      } catch (cause) {
        if (!cancelled) setError(errorMessage(cause, "Factory is unreachable"))
      } finally {
        busy = false
      }
    }

    void tick()
    const timer = window.setInterval(() => void tick(), POLL_INTERVAL_MS)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [polling, repo])

  return error
}
