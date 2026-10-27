import { useRef, useState } from "react"
import { useTabActive } from "@/contexts/TabContext"
import { usePolling } from "@/hooks/usePolling"
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
  const [error, setError] = useState<string | null>(null)
  const cursor = useRef<number | null>(null)

  usePolling(async (signal) => {
    try {
      // The cursor only advances once `refresh` succeeded, so a failed refresh is retried by the next tick.
      if (cursor.current === null) {
        const latest = await fetchLatestFactoryEventSeq()
        if (signal.aborted) return
        await refresh(null)
        cursor.current = latest
      } else {
        const events = await fetchFactoryEvents(repo, cursor.current)
        if (signal.aborted) return
        if (events.length > 0) {
          await refresh(new Set(events.map((event) => event.task_id)))
          cursor.current = events[events.length - 1].seq
        }
      }
      if (!signal.aborted) setError(null)
    } catch (cause) {
      if (!signal.aborted) setError(errorMessage(cause, "Factory is unreachable"))
    }
  }, { intervalMs: POLL_INTERVAL_MS, active, restartKey: repo })

  return error
}
