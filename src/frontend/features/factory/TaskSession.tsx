"use client"

import type { FileLinks } from "@/components/content/LaTeXMarkdown"
import { PiTranscript } from "@/features/pi/PiTranscript"
import type { Transcript } from "@/features/pi/transcript"

/** The session is read-only here: its file links open nothing. */
const NO_FILE_LINKS: FileLinks = { base: "", open: () => {} }

/** A task's coder session as the question and answer cards of a Pi session. */
export function TaskSession({ transcript }: { transcript: Transcript | null }) {
  if (!transcript) return <div className="flex h-full items-center justify-center text-xs text-ink-faint">Loading session</div>
  return <div className="flex h-full w-full flex-col overflow-hidden"><PiTranscript transcript={transcript} fileLinks={NO_FILE_LINKS} /></div>
}
