"use client"

import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react"
import { X } from "lucide-react"
import { ChatInput, type ChatInputHandle } from "@/features/chat/composer/ChatInput"
import type { DraftPicker } from "@/features/chat/composer/CompletionMenu"
import { stageChip, useComposerChips } from "@/features/chat/composer/composerChips"
import { fetchPiFiles } from "@/lib/backend/api"
import type { Attachment } from "@/lib/domain/types"
import { composerAction, piCompletions, type ComposerAction } from "./composerInput"
import type { ExtensionDialog, ImageContent, PiCommand } from "./types"
import type { Notice, QueueMode } from "./usePiSession"

const CONFIRM_OPTIONS = ["Yes", "No"]

/** One composer send: what it does, its images, and how it queues while a run streams (Enter steers, Alt+Enter follows up). */
export interface PiSubmit {
  action: ComposerAction
  images: ImageContent[]
  mode: QueueMode
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
export async function imageOf(attachment: Attachment): Promise<ImageContent> {
  const blob = await (await fetch(attachment.url)).blob()
  const reader = new FileReader()
  const { promise, resolve, reject } = Promise.withResolvers<string>()
  reader.onload = () => resolve(String(reader.result))
  reader.onerror = () => reject(reader.error)
  reader.readAsDataURL(blob)
  const dataUrl = await promise
  return { type: "image", data: dataUrl.slice(dataUrl.indexOf(",") + 1), mimeType: attachment.mime }
}

/** The repo's files for `@` completion, unranked (the composer ranks them), empty until loaded or when the list fails. */
function useRepoFiles(repo: string): string[] {
  const [files, setFiles] = useState<{ repo: string; paths: string[] } | null>(null)
  useEffect(() => {
    let cancelled = false
    fetchPiFiles(repo).then((paths) => cancelled || setFiles({ repo, paths }), () => cancelled || setFiles({ repo, paths: [] }))
    return () => {
      cancelled = true
    }
  }, [repo])
  return files?.repo === repo ? files.paths : []
}

interface PiComposerProps {
  sessionKey: string
  /** Repo under `~/git` whose files `@` completes. */
  repo: string
  running: boolean
  disabled: boolean
  commands: PiCommand[]
  dialog: ExtensionDialog | undefined
  queue: { steering: string[]; followUp: string[] }
  notices: Notice[]
  statusLines: Record<string, string>
  editorText: { text: string } | null
  onSubmit: (submit: PiSubmit) => void
  onAbort: () => Promise<string[]>
  onAnswerDialog: (id: string, answer: { value: string } | { confirmed: boolean } | { cancelled: true }) => void
}

/**
 * gigachad's composer driving Pi: `/` lists Pi's commands, `@` the repo's files, `!cmd`/`!!cmd` run in the shell, `/compact` compacts.
 *
 * - While a run streams Enter steers, Alt+Enter queues a follow-up, Esc (or Stop) aborts and takes queued messages back into the draft
 * - Composer chips join prompts, a `!` command leaves them staged for the next message
 * - `select`/`confirm` dialogs pick in the composer's menu, `input`/`editor` dialogs take the next send as their answer
 * - Queued steering and follow-ups, extension status lines and notices sit above it
 */
export function PiComposer({ sessionKey, repo, running, disabled, commands, dialog, queue, notices, statusLines, editorText, onSubmit, onAbort, onAnswerDialog }: PiComposerProps) {
  const inputRef = useRef<ChatInputHandle>(null)
  const draftKey = `pi:${sessionKey}`
  const chips = useComposerChips(draftKey)
  const files = useRepoFiles(repo)
  const complete = useMemo(() => piCompletions(commands, files), [commands, files])
  const picker = useMemo(() => dialogPicker(dialog, onAnswerDialog), [dialog, onAnswerDialog])
  const textDialog = dialog && (dialog.method === "input" || dialog.method === "editor") ? dialog : undefined
  /** Set while an Alt+Enter keydown passes through `ChatInput`, so its send queues a follow-up. */
  const followUpRef = useRef(false)

  useEffect(() => {
    if (editorText) inputRef.current?.updateText(() => editorText.text)
  }, [editorText])

  useEffect(() => {
    if (textDialog?.method === "editor" && textDialog.prefill) inputRef.current?.updateText(() => textDialog.prefill ?? "")
  }, [textDialog])

  const handleSend = async (text: string, attachments: Attachment[]) => {
    const mode: QueueMode = followUpRef.current ? "followUp" : "steer"
    if (textDialog) return onAnswerDialog(textDialog.id, { value: text })
    const staged = chips.chips
    const action = composerAction(text, staged.map((chip) => chip.text))
    // `ChatInput` clears the chips right after this send: a command did not use them, so they wait for the next message
    if (action.kind === "bash" && staged.length > 0) queueMicrotask(() => staged.forEach((chip) => stageChip(draftKey, chip)))
    const images = action.kind === "prompt" ? await Promise.all(attachments.filter((attachment) => attachment.mime.startsWith("image/")).map(imageOf)) : []
    onSubmit({ action, images, mode })
  }

  const handleAbort = async () => {
    const restored = await onAbort()
    if (restored.length > 0) inputRef.current?.updateText((current) => [...restored, current].filter(Boolean).join("\n\n"))
  }

  // Before `ChatInput`'s own keys: open menus and dialogs take Escape first through their escape layers, so a free Escape aborts the run
  const handleKeyDownCapture = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.nativeEvent.isComposing) return
    if (event.key === "Enter" && event.altKey && !event.shiftKey) followUpRef.current = true
    if (event.key === "Escape" && running) {
      event.preventDefault()
      event.stopPropagation()
      void handleAbort()
    }
  }

  const queued = [...queue.steering.map((text) => ({ text, kind: "Steer" })), ...queue.followUp.map((text) => ({ text, kind: "Follow-up" }))]
  const statuses = Object.entries(statusLines)

  return (
    <div className="shrink-0 px-4 pb-4" onKeyDownCapture={handleKeyDownCapture} onKeyDown={() => { followUpRef.current = false }}>
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
        chatId={draftKey}
        draftKey={draftKey}
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
