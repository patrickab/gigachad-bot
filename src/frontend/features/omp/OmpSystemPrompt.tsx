"use client"

import { useRef, useState } from "react"
import { ScrollText } from "lucide-react"
import { Dialog } from "@/components/ui/Dialog"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"
import { fetchOmpSystemPrompt } from "@/lib/backend/api"
import type { OmpView } from "@/lib/domain/types"
import { ErrorText } from "@/components/ui/ErrorText"
import { errorMessage } from "@/lib/utils"

type PromptState =
  | { status: "loading" }
  | { status: "loaded"; prompt: string }
  | { status: "not-running" }
  | { status: "failed"; message: string }

/** Top bar button showing the system prompt of the omp session in view. Only a running session has one, omp-web never starts a session for it. */
export function OmpSystemPrompt({ view }: { view: OmpView }) {
  const [state, setState] = useState<PromptState | null>(null)
  const requestRef = useRef(0)
  if (view.kind !== "session") return null
  const sessionId = view.sessionId

  // Only the latest opening applies, a close in between drops its answer.
  const open = () => {
    const request = ++requestRef.current
    const settle = (next: PromptState) => { if (request === requestRef.current) setState(next) }
    setState({ status: "loading" })
    fetchOmpSystemPrompt(sessionId).then(
      (result) => settle(result.running && result.system_prompt ? { status: "loaded", prompt: result.system_prompt } : { status: "not-running" }),
      (error: unknown) => settle({ status: "failed", message: errorMessage(error, "Could not load the system prompt") }),
    )
  }
  const close = () => {
    requestRef.current++
    setState(null)
  }

  return <>
    <button type="button" aria-label="Show system prompt" onClick={open} className="rounded p-1 text-ink-subtle transition-colors hover:bg-hover hover:text-ink">
      <ScrollText className="h-4 w-4" />
    </button>
    <Dialog open={state !== null} onClose={close} icon={ScrollText} title="System prompt" widthClassName="max-w-3xl">
      <div className="max-h-[70vh] overflow-y-auto px-5 pb-5 text-sm">
        {state?.status === "loading" && <p className="text-xs text-ink-subtle">Loading…</p>}
        {state?.status === "not-running" && <p className="text-xs text-ink-subtle">This session is not running. Its system prompt shows once it runs, after its next message.</p>}
        {state?.status === "failed" && <ErrorText>{state.message}</ErrorText>}
        {state?.status === "loaded" && <LaTeXMarkdown content={state.prompt} compact />}
      </div>
    </Dialog>
  </>
}
