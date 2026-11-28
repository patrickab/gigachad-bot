"use client"

import type { FileLinks } from "@/components/content/LaTeXMarkdown"
import { useLoaded } from "@/hooks/useLoaded"
import { fetchPiSessionEntries, type PiSessionInfo } from "@/lib/backend/api"
import { PiTranscript } from "./PiTranscript"
import { transcriptOfEntries, type Transcript } from "./transcript"

/** Turns a preview shows: the window fits a few, so only the last ones are read, never the whole file. */
const PREVIEW_TURNS = 6
/** How many of the most recently used sessions are read ahead, so their previews show at once. */
const PREFETCH_COUNT = 8
/** A preview is a picture: its file links open nothing. */
const NO_FILE_LINKS: FileLinks = { base: "", open: () => {} }

/** The preview transcript per session, the read for its file version, kept until the file changes. */
const previews = new Map<string, { modifiedAt: string; transcript: Promise<Transcript> }>()

/** The preview transcript of `session`, read once per file version; a failed read is dropped so the next call retries. */
function loadPreview({ sessionId, modifiedAt }: Pick<PiSessionInfo, "sessionId" | "modifiedAt">): Promise<Transcript> {
  const cached = previews.get(sessionId)
  if (cached?.modifiedAt === modifiedAt) return cached.transcript
  const transcript = fetchPiSessionEntries(sessionId, { turns: PREVIEW_TURNS }).then(transcriptOfEntries)
  previews.set(sessionId, { modifiedAt, transcript })
  transcript.catch(() => { if (previews.get(sessionId)?.transcript === transcript) previews.delete(sessionId) })
  return transcript
}

/**
 * Read the previews of the `PREFETCH_COUNT` most recently used of `sessions` (newest first) ahead, while the browser idles.
 * A running session's is left out: its file changes with every refresh, so its preview loads when shown.
 */
export function prefetchPiSessionPreviews(sessions: readonly PiSessionInfo[]): () => void {
  const read = () => sessions.slice(0, PREFETCH_COUNT).forEach((session) => { if (!session.live?.running) void loadPreview(session).catch(() => {}) })
  if (typeof requestIdleCallback === "undefined") {
    const timer = setTimeout(read, 0)
    return () => clearTimeout(timer)
  }
  const handle = requestIdleCallback(read)
  return () => cancelIdleCallback(handle)
}

/** The preview transcript of the session `key` names, `<sessionId>\n<modifiedAt>`. */
const loadPreviewOfKey = (key: string) => {
  const [sessionId, modifiedAt] = key.split("\n")
  return loadPreview({ sessionId, modifiedAt })
}

/**
 * A picker preview of a session that need not be open: its last turns read from the session file, shown like its tab.
 *
 * - The turn cards keep their normal size and the card width stored app-wide (`useColumnWidth`), centered in the window
 * - Nothing attaches and nothing streams; the picker's preview window gives the background
 * - Reads go through one cache per file version, `prefetchPiSessionPreviews` fills it ahead for the recent sessions
 */
export function PiSessionPreview({ session }: { session: PiSessionInfo }) {
  // Empty while it is read or when the read failed
  const transcript = useLoaded(`${session.sessionId}\n${session.modifiedAt}`, loadPreviewOfKey).value
  return <div className="flex h-full w-full flex-col overflow-hidden">
    {transcript && <PiTranscript transcript={transcript} fileLinks={NO_FILE_LINKS} readOnly />}
  </div>
}
