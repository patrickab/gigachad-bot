import { MessageAttachments } from "@/features/chat/messages/MessageAttachments"
import type { HeldMessage } from "./outbox"
import { toolImageAttachments } from "./toolImages"

const LABELS = { steer: "Steer", followUp: "Follow-up" }

function statusOf(entry: HeldMessage): string {
  if (entry.state === "failed") return "Not sent"
  if (entry.state === "queued") return `Queued · ${LABELS[entry.mode]}`
  return "Sending"
}

/** Messages sent during a run that pi-host has not confirmed yet, below the transcript. Not sent ones offer Retry and Delete. */
export function HeldMessages({ held, onRetry, onDelete }: { held: readonly HeldMessage[]; onRetry: (id: string) => void; onDelete: (id: string) => void }) {
  if (held.length === 0) return null
  return (
    <ul aria-label="Held messages" className="flex flex-col items-end gap-2 px-4 pb-3">
      {held.map((entry) => (
        <li key={entry.id} className="max-w-[85%] rounded-xl bg-paper px-3 py-2 text-sm text-ink-muted">
          {entry.text && <p className="whitespace-pre-wrap break-words">{entry.text}</p>}
          {entry.images.length > 0 && <MessageAttachments attachments={toolImageAttachments("image", entry.images)} onClick={() => {}} />}
          <p className={`mt-1 flex items-center gap-2 text-xs ${entry.state === "failed" ? "text-danger" : "text-ink-faint"}`}>
            <span>{statusOf(entry)}</span>
            {entry.state === "failed" && (
              <>
                <button type="button" onClick={() => onRetry(entry.id)} className="rounded hover:text-ink focus-visible:outline focus-visible:outline-1">Retry</button>
                <button type="button" onClick={() => onDelete(entry.id)} className="rounded hover:text-ink focus-visible:outline focus-visible:outline-1">Delete</button>
              </>
            )}
          </p>
        </li>
      ))}
    </ul>
  )
}
