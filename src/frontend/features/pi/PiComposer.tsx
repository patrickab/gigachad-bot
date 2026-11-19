"use client"

import { useEffect, useMemo, useRef } from "react"
import { MessageSquareText, Puzzle, Sparkles, X, type LucideIcon } from "lucide-react"
import { ChatInput, type ChatInputHandle } from "@/features/chat/composer/ChatInput"
import type { CompleteDraft, DraftPicker } from "@/features/chat/composer/CompletionMenu"
import { fuzzyScore } from "@/lib/fuzzy"
import type { Attachment } from "@/lib/domain/types"
import type { ExtensionDialog, PiCommand } from "./types"
import type { Notice } from "./usePiSession"

const COMMAND_ICONS: Record<PiCommand["source"], LucideIcon> = { extension: Puzzle, prompt: MessageSquareText, skill: Sparkles }
const CONFIRM_OPTIONS = ["Yes", "No"]

/** `/` suggestions while the draft is one word starting with `/`, ranked by fuzzy score. */
function slashCompletions(commands: PiCommand[]): CompleteDraft {
  return async (text) => {
    if (!text.startsWith("/") || /\s/.test(text)) return []
    const query = text.slice(1).toLowerCase()
    const ranked = commands
      .map((command) => ({ command, score: query ? fuzzyScore(command.name, query) : 0 }))
      .filter((entry): entry is { command: PiCommand; score: number } => entry.score !== null)
      .sort((left, right) => right.score - left.score || left.command.name.localeCompare(right.command.name))
    return ranked.map(({ command }) => {
      const insert = `/${command.name} `
      return { label: `/${command.name}`, detail: command.description, icon: COMMAND_ICONS[command.source], from: 0, to: text.length, insert, caretOffset: insert.length }
    })
  }
}

/** A `select` or `confirm` dialog as the composer's draft picker, answered with the picked option. */
function dialogPicker(dialog: ExtensionDialog | undefined, answer: PiComposerProps["onAnswerDialog"]): DraftPicker | undefined {
  if (!dialog || (dialog.method !== "select" && dialog.method !== "confirm")) return undefined
  const options = dialog.method === "select" ? dialog.options : CONFIRM_OPTIONS
  return {
    label: dialog.title,
    items: (query) => options
      .filter((option) => option.toLowerCase().includes(query.trim().toLowerCase()))
      .map((option) => ({ key: option, label: option })),
    onPick: (option) => answer(dialog.id, dialog.method === "confirm" ? { confirmed: option === "Yes" } : { value: option }),
    onCancel: () => answer(dialog.id, { cancelled: true }),
  }
}

/** `attachment` (an image) as a Pi image block. */
export async function imageOf(attachment: Attachment): Promise<{ type: "image"; data: string; mimeType: string }> {
  const blob = await (await fetch(attachment.url)).blob()
  const reader = new FileReader()
  const { promise, resolve, reject } = Promise.withResolvers<string>()
  reader.onload = () => resolve(String(reader.result))
  reader.onerror = () => reject(reader.error)
  reader.readAsDataURL(blob)
  const dataUrl = await promise
  return { type: "image", data: dataUrl.slice(dataUrl.indexOf(",") + 1), mimeType: attachment.mime }
}

interface PiComposerProps {
  sessionKey: string
  running: boolean
  disabled: boolean
  commands: PiCommand[]
  dialog: ExtensionDialog | undefined
  queue: { steering: string[]; followUp: string[] }
  notices: Notice[]
  statusLines: Record<string, string>
  editorText: { text: string } | null
  onSend: (message: string, images: { type: "image"; data: string; mimeType: string }[]) => void
  onAbort: () => Promise<string[]>
  onAnswerDialog: (id: string, answer: { value: string } | { confirmed: boolean } | { cancelled: true }) => void
}

/**
 * gigachad's composer driving Pi: `/` lists Pi's commands, Enter while running steers, Stop takes queued messages back into the draft.
 *
 * - `select`/`confirm` dialogs pick in the composer's menu, `input`/`editor` dialogs take the next send as their answer
 * - Queued steering and follow-ups, extension status lines and notices sit above it
 */
export function PiComposer({ sessionKey, running, disabled, commands, dialog, queue, notices, statusLines, editorText, onSend, onAbort, onAnswerDialog }: PiComposerProps) {
  const inputRef = useRef<ChatInputHandle>(null)
  const complete = useMemo(() => slashCompletions(commands), [commands])
  const picker = useMemo(() => dialogPicker(dialog, onAnswerDialog), [dialog, onAnswerDialog])
  const textDialog = dialog && (dialog.method === "input" || dialog.method === "editor") ? dialog : undefined

  useEffect(() => {
    if (editorText) inputRef.current?.updateText(() => editorText.text)
  }, [editorText])

  useEffect(() => {
    if (textDialog?.method === "editor" && textDialog.prefill) inputRef.current?.updateText(() => textDialog.prefill ?? "")
  }, [textDialog])

  const handleSend = async (text: string, attachments: Attachment[]) => {
    if (textDialog) return onAnswerDialog(textDialog.id, { value: text })
    const images = await Promise.all(attachments.filter((attachment) => attachment.mime.startsWith("image/")).map(imageOf))
    onSend(text, images)
  }

  const handleAbort = async () => {
    const restored = await onAbort()
    if (restored.length > 0) inputRef.current?.updateText((current) => [...restored, current].filter(Boolean).join("\n\n"))
  }

  const queued = [...queue.steering.map((text) => ({ text, kind: "Steer" })), ...queue.followUp.map((text) => ({ text, kind: "Follow-up" }))]
  const statuses = Object.entries(statusLines)

  return (
    <div className="shrink-0 px-4 pb-4">
      <div className="mx-auto mb-2 flex w-full max-w-2xl flex-col gap-1 text-xs">
        {notices.map((notice) => (
          <p key={notice.id} role="status" className={notice.tone === "error" ? "text-danger" : notice.tone === "warning" ? "text-warning" : "text-ink-muted"}>{notice.text}</p>
        ))}
        {queued.map((entry, index) => (
          <p key={`${entry.kind}-${index}`} className="truncate text-ink-muted"><span className="text-ink-faint">{entry.kind}</span> {entry.text}</p>
        ))}
        {statuses.length > 0 && <p className="truncate text-ink-faint">{statuses.map(([key, text]) => <span key={key} className="mr-3">{text}</span>)}</p>}
        {textDialog && (
          <div className="flex items-center gap-2 text-ink-muted">
            <span className="truncate">{textDialog.title}</span>
            <button type="button" aria-label="Cancel" onClick={() => onAnswerDialog(textDialog.id, { cancelled: true })} className="rounded p-0.5 hover:text-ink focus-visible:outline focus-visible:outline-1">
              <X size={12} aria-hidden="true" />
            </button>
          </div>
        )}
      </div>
      <ChatInput
        ref={inputRef}
        chatId={`pi:${sessionKey}`}
        draftKey={`pi:${sessionKey}`}
        onSend={(text, attachments) => void handleSend(text, attachments)}
        disabled={disabled}
        isStreaming={running}
        sendWhileStreaming
        onCancel={() => void handleAbort()}
        toolsMenu={false}
        placeholder={textDialog && textDialog.method === "input" ? textDialog.placeholder ?? "Message" : "Message"}
        complete={textDialog ? undefined : complete}
        picker={picker}
      />
    </div>
  )
}
