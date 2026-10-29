"use client"

import { forwardRef, useCallback, useImperativeHandle, useRef, useState, type ElementType } from "react"
import { motion } from "framer-motion"
import { Mic, Pencil } from "lucide-react"
import { cn } from "@/lib/utils"
import { useSpeechInput } from "@/hooks/useSpeechInput"
import type { Attachment } from "@/lib/domain/types"
import { AttachmentPreview } from "@/features/chat/preview/AttachmentPreview"
import { AttachmentChip, type ContextAttachment } from "./AttachmentChip"
import { CompletionMenu, useCompletions, useDraftPicker, type CompleteDraft, type DraftPicker } from "./CompletionMenu"
import { AttachMenu, ICON_BUTTON, SendButton, ToolsMenu } from "./ComposerControls"
import { DrawingAttach, type DrawingMode } from "./DrawingAttach"
import { useDraftAttachments } from "./useDraftAttachments"

interface ChatInputProps {
  chatId: string
  onSend: (text: string, attachments: Attachment[]) => void
  disabled?: boolean
  extracting?: boolean
  isStreaming?: boolean
  onCancel?: () => void
  slug?: string | null
  /** gigachad's chat tools menu, off where another agent owns the tools (omp tabs). */
  toolsMenu?: boolean
  placeholder?: string
  /** Suggestions while typing (omp commands and files), none without. */
  complete?: CompleteDraft
  /** A choice the draft filters (an omp extension's `select`), shown instead of suggestions. Enter picks, never sends, while set. */
  picker?: DraftPicker
  /** Mode note under the draft, e.g. that `!ls` runs in a shell, null for none. */
  hint?: (text: string, attachmentCount: number) => string | null
  /** Attachments of earlier messages, shown as chips before the draft's own. */
  context?: ChatContext
  /** Extra entries of the "+" menu after the built-in ones. */
  attachMenuItems?: AttachMenuItem[]
}

/** Attachments already in the chat with their context actions. */
export interface ChatContext {
  attachments: ContextAttachment[]
  onToggleActive: (messageIndex: number, attachmentName: string) => void
  onRemove: (messageIndex: number, attachmentName: string) => void
  onOpen: (messageIndex: number, attachment: Attachment) => void
}

export interface AttachMenuItem {
  label: string
  icon: ElementType
  onClick: () => void
}

export interface ChatInputHandle {
  addAttachment: (att: Attachment) => void
  /** Replace the draft text with `update(current)` and focus the input. */
  updateText: (update: (current: string) => string) => void
  /** Upload and attach files, e.g. images pasted into an omp tab. */
  addFiles: (files: File[]) => void
  focus: () => void
}

export const ChatInput = forwardRef<ChatInputHandle, ChatInputProps>(function ChatInput({
  chatId,
  onSend,
  disabled,
  extracting,
  isStreaming,
  onCancel,
  slug = null,
  toolsMenu = true,
  placeholder = "Message",
  complete,
  picker,
  hint,
  context,
  attachMenuItems = [],
}, ref) {
  const drafts = useDraftAttachments(chatId, slug)
  const { attachments, upload } = drafts
  const [text, setText] = useState("")
  const [drawing, setDrawing] = useState<DrawingMode | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const adjustHeight = useCallback(() => {
    const el = textareaRef.current
    if (!el) return
    // Reset first so scrollHeight can shrink
    el.style.height = "0"
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }, [])
  const adjustHeightNextFrame = useCallback(() => { requestAnimationFrame(() => adjustHeight()) }, [adjustHeight])

  const replaceText = useCallback((next: string) => {
    setText(next)
    adjustHeightNextFrame()
  }, [adjustHeightNextFrame])
  const completions = useCompletions(complete, textareaRef, replaceText)
  const focusDraft = useCallback(() => textareaRef.current?.focus(), [])
  const clearDraft = useCallback(() => replaceText(""), [replaceText])
  const draftPicker = useDraftPicker(picker, text, focusDraft, clearDraft)
  const { isListening, toggleListening } = useSpeechInput(text, setText, adjustHeightNextFrame)

  useImperativeHandle(ref, () => ({
    addAttachment: drafts.add,
    updateText: (update) => {
      setText(update)
      requestAnimationFrame(() => { adjustHeight(); textareaRef.current?.focus() })
    },
    addFiles: (files) => { for (const file of files) upload(file, file.name) },
    focus: () => textareaRef.current?.focus(),
  }), [adjustHeight, drafts.add, upload])

  const handleSubmit = () => {
    const trimmed = text.trim()
    if (!trimmed && attachments.length === 0) return
    completions.close()
    onSend(trimmed || "", attachments)
    setText("")
    drafts.clear()
    adjustHeightNextFrame()
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (draftPicker.handleKey(e) || completions.handleKey(e, text)) return
    if (e.key === "Enter" && !e.shiftKey && !isStreaming && !extracting) { e.preventDefault(); handleSubmit() }
  }

  // Only the first pasted image is attached
  const handlePaste = (e: React.ClipboardEvent) => {
    for (const item of e.clipboardData.items) {
      if (item.type.startsWith("image/")) {
        e.preventDefault()
        const blob = item.getAsFile()
        if (!blob) continue
        upload(blob, blob.name || `pasted-image.${item.type.split("/")[1] || "png"}`)
        break
      }
    }
  }

  const canSend = !picker && (text.trim().length > 0 || attachments.length > 0)
  const hintText = hint?.(text, attachments.length) ?? null
  const menuItems: (AttachMenuItem & { iconClassName?: string })[] = [
    ...attachMenuItems,
    { label: "Drawing", icon: Pencil, onClick: () => setDrawing("attach") },
    { label: "Drawing (OCR)", icon: Pencil, onClick: () => setDrawing("ocr"), iconClassName: "text-ink" },
  ]

  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="mx-auto w-full max-w-2xl">
      {((context?.attachments.length ?? 0) > 0 || attachments.length > 0) && (
        <div className="relative z-10 mb-2 flex flex-wrap gap-1.5">
          {context?.attachments.map(({ messageIndex, attachment }) => (
            <AttachmentChip
              key={`${messageIndex}-${attachment.name}`}
              attachment={attachment}
              onOpen={() => context.onOpen(messageIndex, attachment)}
              onRemove={() => context.onRemove(messageIndex, attachment.name)}
              onToggleActive={() => context.onToggleActive(messageIndex, attachment.name)}
            />
          ))}
          {attachments.map((attachment) => (
            <AttachmentChip
              key={attachment.name}
              attachment={attachment}
              uploading={drafts.uploadingNames.has(attachment.name)}
              onOpen={() => void drafts.openPreview(attachment)}
              onRemove={() => drafts.remove(attachment.name)}
            />
          ))}
        </div>
      )}

      <div className="relative z-20 rounded-2xl border border-divider/50 px-4 pb-2.5 pt-3.5 transition-all duration-300 bg-surface shadow-[var(--shadow-xl),var(--inner-highlight)] focus-within:border-divider focus-within:shadow-[var(--shadow-xl),var(--inner-highlight-strong)]">
          {picker ? (
            <CompletionMenu
              label={picker.label}
              items={draftPicker.items}
              activeIndex={draftPicker.activeIndex}
              onHover={draftPicker.setActiveIndex}
              onApply={(item) => draftPicker.pick(item.key)}
              emptyText="No matches"
            />
          ) : (
            <CompletionMenu
              items={completions.items}
              activeIndex={completions.activeIndex}
              onHover={completions.setActiveIndex}
              onApply={(item) => completions.apply(text, item)}
            />
          )}
          <textarea
            ref={textareaRef}
            rows={1}
            value={text}
            onChange={(e) => { setText(e.target.value); adjustHeight(); if (!picker) completions.request(e.target.value, e.target.selectionStart) }}
            onKeyDown={handleKeyDown}
            onBlur={completions.close}
            onPaste={handlePaste}
            disabled={disabled && !isStreaming}
            placeholder={placeholder}
            spellCheck={false}
            autoCorrect="off"
            autoCapitalize="off"
            data-gramm="false"
            data-gramm_editor="false"
            data-enable-grammarly="false"
            className="w-full resize-none bg-transparent text-sm leading-6 text-ink placeholder:text-ink-faint outline-none"
          />
          {hintText && <p className="mt-1 text-xs text-ink-muted">{hintText}</p>}
          <div className="mt-2 flex items-center justify-between">
            <div className="-ml-1.5 flex min-w-0 items-center gap-0.5">
              <AttachMenu items={menuItems} disabled={disabled} />
            </div>
            <div className="flex items-center gap-0.5">
              {toolsMenu && <ToolsMenu disabled={disabled} />}
              <button aria-label={isListening ? "Stop listening" : "Start listening"}
                onClick={toggleListening}
                disabled={disabled}
                className={cn(ICON_BUTTON, isListening && "bg-danger-soft text-danger hover:bg-danger-soft hover:text-danger")}
              >
                <Mic className={cn("h-4 w-4", isListening && "animate-pulse")} />
              </button>
              <SendButton isStreaming={isStreaming} extracting={extracting} ready={canSend} disabled={disabled} onCancel={onCancel} onSend={handleSubmit} />
            </div>
          </div>
        </div>
      <DrawingAttach
        mode={drawing}
        chatId={chatId}
        onClose={() => setDrawing(null)}
        onAttach={drafts.add}
        onUpload={(file) => upload(file, file.name)}
      />
      <AttachmentPreview attachment={drafts.preview} onClose={drafts.closePreview} onContentChange={drafts.editPreview} />
    </motion.div>
  )
})
