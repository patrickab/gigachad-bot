// A task's coder session as a transcript. Consumes `GET /api/pi/sessions/<id>/entries` (the persisted session file),
// provides `useTaskSession` (the transcript, read again whenever the task moves).
import { useEffect, useState } from "react"
import { fetchPiSessionEntries } from "@/lib/backend/api"
import { transcriptOfEntries, type Transcript } from "@/features/pi/transcript"
import { errorMessage } from "@/lib/utils"
import { TIMELINE_POLL_MS } from "./useTaskTimeline"

/** Reads of session `sessionId`, one at a time: a read asked for while one is pending is dropped. */
function sessionReader(sessionId: string, show: (transcript: Transcript) => void, report: (error: string) => void): () => void {
  let reading = false
  return () => {
    if (reading) return
    reading = true
    fetchPiSessionEntries(sessionId)
      .then((entries) => { show(transcriptOfEntries(entries)); report("") })
      .catch((cause) => report(errorMessage(cause, "Could not load the session")))
      .finally(() => { reading = false })
  }
}

/**
 * The transcript of the session `sessionId`, null until the first read.
 *
 * - Read again whenever `revision` changes and every `TIMELINE_POLL_MS` while `running`
 * - `sessionId` is fixed per mount: key the owner by task
 * - A failed read keeps the transcript and sets `error`, the next successful one clears it
 */
export function useTaskSession(sessionId: string, revision: string, running: boolean): { transcript: Transcript | null; error: string } {
  const [transcript, setTranscript] = useState<Transcript | null>(null)
  const [error, setError] = useState("")
  const [read] = useState(() => sessionReader(sessionId, setTranscript, setError))

  useEffect(() => { read() }, [read, revision])
  useEffect(() => {
    if (!running) return
    const timer = setInterval(read, TIMELINE_POLL_MS)
    return () => clearInterval(timer)
  }, [read, running])

  return { transcript, error }
}
