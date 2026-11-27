"use client"

import { useCallback, useState } from "react"
import type { FileLinks } from "@/components/content/LaTeXMarkdown"
import { useLoadWhile } from "@/hooks/useLoadWhile"
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

/** Read the previews of the `PREFETCH_COUNT` most recently used of `sessions` (newest first) ahead, while the browser idles. */
export function prefetchPiSessionPreviews(sessions: readonly PiSessionInfo[]): () => void {
  const read = () => sessions.slice(0, PREFETCH_COUNT).forEach((session) => { void loadPreview(session).catch(() => {}) })
  if (typeof requestIdleCallback === "undefined") {
    const timer = setTimeout(read, 0)
    return () => clearTimeout(timer)
  }
  const handle = requestIdleCallback(read)
  return () => cancelIdleCallback(handle)
}

/** `session`'s preview transcript, null while it is read or when the read failed (the window shows empty). */
function usePreview({ sessionId, modifiedAt }: PiSessionInfo): Transcript | null {
  const [shown, setShown] = useState<{ key: string; transcript: Transcript } | null>(null)
  const key = `${sessionId}:${modifiedAt}`
  // The answer carries the key it was read for, so one delivered just after the session changed never passes for the next.
  const load = useCallback(() => loadPreview({ sessionId, modifiedAt }).then((transcript) => ({ key: `${sessionId}:${modifiedAt}`, transcript })), [sessionId, modifiedAt])
  useLoadWhile(true, load, setShown)
  return shown?.key === key ? shown.transcript : null
}

/**
 * A picker preview of a session that need not be open: its last turns read from the session file, shown like its tab.
 *
 * - The turn cards keep their normal size and the card width stored app-wide (`useColumnWidth`), centered in the window
 * - Nothing attaches and nothing streams; the picker's preview window gives the background
 * - Reads go through one cache per file version, `prefetchPiSessionPreviews` fills it ahead for the recent sessions
 */
export function PiSessionPreview({ session }: { session: PiSessionInfo }) {
  const transcript = usePreview(session)
  return <div className="flex h-full w-full flex-col overflow-hidden">
    {transcript && <PiTranscript transcript={transcript} fileLinks={NO_FILE_LINKS} readOnly />}
  </div>
}
