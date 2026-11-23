"use client"

import { useState } from "react"
import { Square, X } from "lucide-react"
import { LaTeXMarkdown, type FileLinks } from "@/components/content/LaTeXMarkdown"
import { IconButton } from "@/components/ui/IconButton"
import type { BtwThread } from "./btw"

export interface BtwActions {
  followUp: (id: string, question: string) => void
  cancel: (id: string) => void
  dismiss: (id: string) => void
}

function Exchange({ question, answer, streaming, error, fileLinks }: { question: string; answer: string; streaming: boolean; error?: string; fileLinks?: FileLinks }) {
  return (
    <div className="flex flex-col gap-1">
      <p className="break-words text-ink-muted">{question}</p>
      {answer && <LaTeXMarkdown content={answer} streaming={streaming} fileLinks={fileLinks} />}
      {error && <p role="status" className="text-danger">{error}</p>}
    </div>
  )
}

function FollowUpInput({ onSend, onClose, disabled }: { onSend: (question: string) => void; onClose: () => void; disabled: boolean }) {
  const [draft, setDraft] = useState("")
  return (
    <input
      aria-label="Follow-up"
      value={draft}
      disabled={disabled}
      placeholder="Follow up"
      onChange={(event) => setDraft(event.target.value)}
      onKeyDown={(event) => {
        if (event.nativeEvent.isComposing) return
        if (event.key === "Escape") {
          event.preventDefault()
          event.stopPropagation()
          onClose()
        } else if (event.key === "Enter" && !event.shiftKey && draft.trim()) {
          event.preventDefault()
          onSend(draft.trim())
          setDraft("")
        }
      }}
      className="w-full rounded-md bg-transparent px-1 py-1 text-ink outline-none placeholder:text-ink-faint focus-visible:outline-1 focus-visible:outline-ink-muted disabled:opacity-40"
    />
  )
}

/**
 * The `/btw` side questions above the composer: the newest thread (question, streaming answer, follow-ups, a follow-up input,
 * Stop while it streams, close to dismiss it), older ones collapsed. A thin client of the `btw` extension's state.
 */
export function BtwPanel({ threads, actions, fileLinks }: { threads: BtwThread[]; actions: BtwActions; fileLinks?: FileLinks }) {
  if (threads.length === 0) return null
  const current = threads[threads.length - 1]
  const older = threads.slice(0, -1).reverse()
  const streaming = current.status === "streaming"
  const last = current.followUps.at(-1)
  return (
    <section aria-label="Side question" className="mx-auto mb-2 flex max-h-[40vh] w-full max-w-2xl flex-col gap-2 overflow-y-auto rounded-xl bg-paper px-3 py-2 text-sm">
      <div className="flex items-start gap-2">
        <div className="min-w-0 flex-1">
          <Exchange question={current.question} answer={current.answer} streaming={streaming && !last} error={!last && current.status === "error" ? current.error : undefined} fileLinks={fileLinks} />
        </div>
        {streaming && <IconButton icon={Square} label="Cancel side question" onClick={() => actions.cancel(current.id)} />}
        <IconButton icon={X} label="Close side question" onClick={() => actions.dismiss(current.id)} />
      </div>
      {current.followUps.map((followUp, index) => (
        <Exchange key={`${followUp.createdAt}-${index}`} question={followUp.question} answer={followUp.answer} streaming={followUp.status === "streaming"} error={followUp.status === "error" ? followUp.error : undefined} fileLinks={fileLinks} />
      ))}
      <FollowUpInput key={current.id} disabled={streaming} onSend={(question) => actions.followUp(current.id, question)} onClose={() => actions.dismiss(current.id)} />
      {older.length > 0 && (
        <details className="text-xs text-ink-subtle">
          <summary className="cursor-pointer select-none hover:text-ink">Earlier ({older.length})</summary>
          <ul className="mt-1 flex flex-col gap-1">
            {older.map((thread) => (
              <li key={thread.id}>
                <details>
                  <summary className="cursor-pointer truncate hover:text-ink">{thread.question}</summary>
                  <div className="mt-1 flex flex-col gap-2 text-sm">
                    <Exchange question={thread.question} answer={thread.answer} streaming={thread.status === "streaming" && thread.followUps.length === 0} fileLinks={fileLinks} />
                    {thread.followUps.map((followUp, index) => (
                      <Exchange key={index} question={followUp.question} answer={followUp.answer} streaming={followUp.status === "streaming"} error={followUp.error} fileLinks={fileLinks} />
                    ))}
                    <div className="flex justify-end">
                      <IconButton icon={X} label={`Close ${thread.question}`} onClick={() => actions.dismiss(thread.id)} />
                    </div>
                  </div>
                </details>
              </li>
            ))}
          </ul>
        </details>
      )}
    </section>
  )
}
