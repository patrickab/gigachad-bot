"use client"

import { forwardRef, useCallback, useImperativeHandle, useLayoutEffect, useRef, useState, type ElementType, type ReactNode } from "react"
import { motion } from "framer-motion"
import { Mic, Pencil } from "lucide-react"
import { cn } from "@/lib/utils"
import { useSpeechInput } from "@/hooks/useSpeechInput"
import type { Attachment } from "@/lib/domain/types"
import { useElementSize } from "@/hooks/useElementSize"
import { AttachmentPreview } from "@/features/chat/preview/AttachmentPreview"
import { AttachmentChip, InactiveAttachments, type ContextAttachment } from "./AttachmentChip"
import { CompletionMenu, useCompletions, useDraftPicker, type CompleteDraft, type DraftPicker } from "./CompletionMenu"
import { COMPOSER_ICON_BUTTON } from "@/components/ui/Composer"
import { AttachMenu, SendButton, ToolsMenu } from "./ComposerControls"
import { DrawingAttach, type DrawingMode } from "./DrawingAttach"
import { useDraftAttachments } from "./useDraftAttachments"
import { useComposerDraft } from "./useComposerDraft"
import { largePasteChip, useComposerChips, withChips } from "./composerChips"
import { TextChip } from "./TextChip"

interface ChatInputProps {
  chatId: string
  /** Conversation the unsent text and staged chips are kept for (`useComposerDraft`, `composerChips`), none keeps them in memory only. */
  draftKey?: string | null
  /** A send that settles `false` (or rejects) was refused: the draft, chips and attachments come back into the composer. */
  onSend: (text: string, attachments: Attachment[]) => void | boolean | Promise<boolean | void>
  disabled?: boolean
  extracting?: boolean
  isStreaming?: boolean
  onCancel?: () => void
  /** Enter and the send button submit while streaming (the agent queues the message), the button stops only with an empty draft. */
  sendWhileStreaming?: boolean
  slug?: string | null
  /** gigachad's chat tools menu, off where another agent owns the tools (Pi tabs). */
  toolsMenu?: boolean
  placeholder?: string
  /** Suggestions while typing (Pi commands and files), none without. */
  complete?: CompleteDraft
  /** A choice the draft filters (a Pi extension's `select`), shown instead of suggestions. Enter picks, never sends, while set. */
  picker?: DraftPicker
  /** Mode note under the draft. */
  hint?: string
  /** Attachments of earlier messages, shown as chips before the draft's own. */
  context?: ChatContext
  /** Extra entries of the "+" menu after the built-in ones. */
  attachMenuItems?: AttachMenuItem[]
  /** Extra controls of the action row after the "+" menu, e.g. the folder a Pi session starts in. */
  leading?: ReactNode
  /** Takes pasted, dropped and attached videos (a Pi tab stores them for an `@` mention), without it they upload like other files. */
  onVideo?: (video: File) => void
  /** Refuses attached, pasted and dropped files it returns `false` for (videos go to `onVideo` first), they are reported to `onRefused` and not uploaded. */
  accept?: (file: File) => boolean
  onRefused?: (files: File[]) => void
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
  /** Upload and attach files, e.g. a canvas screenshot put into a Pi chat. Videos go to `onVideo` when given. */
  addFiles: (files: File[]) => void
  focus: () => void
}

export const ChatInput = forwardRef<ChatInputHandle, ChatInputProps>(function ChatInput({
  chatId,
  draftKey = null,
  onSend,
  disabled,
  extracting,
  isStreaming,
  onCancel,
  sendWhileStreaming = false,
  slug = null,
  toolsMenu = true,
  placeholder = "Message",
  complete,
  picker,
  hint,
  context,
  attachMenuItems = [],
  leading,
  onVideo,
  accept,
  onRefused,
}, ref) {
  const drafts = useDraftAttachments(chatId, slug)
  const { attachments, upload } = drafts
  const [text, setText] = useComposerDraft(draftKey)
  const chips = useComposerChips(draftKey)
  const [drawing, setDrawing] = useState<DrawingMode | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Fit every text and width change: a restored draft, a panel springing open from zero width. Reset first so scrollHeight can shrink
  const { width: textareaWidth } = useElementSize(textareaRef)
  useLayoutEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = "0"
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }, [text, textareaWidth])

  const completions = useCompletions(complete, textareaRef, setText)
  const focusDraft = useCallback(() => textareaRef.current?.focus(), [])
  const clearDraft = useCallback(() => setText(""), [setText])
  const draftPicker = useDraftPicker(picker, text, focusDraft, clearDraft)
  const { isListening, toggleListening } = useSpeechInput(text, setText)

  /** Videos go to `onVideo` when given, other files upload as attachments. */
  const takeFiles = useCallback((files: File[]) => {
    const refused: File[] = []
    for (const file of files) {
      if (onVideo && file.type.startsWith("video/")) onVideo(file)
      else if (accept && !accept(file)) refused.push(file)
      else upload(file, file.name)
    }
    if (refused.length > 0) onRefused?.(refused)
  }, [onVideo, accept, onRefused, upload])

  useImperativeHandle(ref, () => ({
    addAttachment: drafts.add,
    updateText: (update) => {
      setText(update)
      requestAnimationFrame(() => textareaRef.current?.focus())
    },
    addFiles: takeFiles,
    focus: () => textareaRef.current?.focus(),
  }), [drafts.add, setText, takeFiles])

  // Staged chips go after the typed text, verbatim
  const handleSubmit = () => {
    const trimmed = text.trim()
    if (!trimmed && attachments.length === 0 && chips.chips.length === 0) return
    completions.close()
    const sentChips = chips.chips
    const sentAttachments = attachments
    const result = onSend(withChips(trimmed, sentChips), sentAttachments)
    setText("")
    drafts.clear()
    chips.clear()
    if (result === undefined || result === true) return
    // Refused: put the draft back, ahead of anything typed meanwhile
    const restore = () => {
      setText((current) => (current ? `${trimmed}\n\n${current}` : trimmed))
      sentChips.forEach((chip) => chips.stage({ label: chip.label, text: chip.text }))
      sentAttachments.forEach(drafts.add)
    }
    if (result === false) restore()
    else void Promise.resolve(result).then((ok) => { if (ok === false) restore() }, restore)
  }

  // Escape layers (completion menu, dialogs) take Escape first, a free Escape leaves the draft
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (draftPicker.handleKey(e) || completions.handleKey(e, text)) return
    if (e.key === "Escape" && !e.nativeEvent.isComposing) { e.preventDefault(); e.currentTarget.blur() }
    if (e.key === "Enter" && !e.shiftKey && !submitBlocked) { e.preventDefault(); handleSubmit() }
  }

  // Videos go to `onVideo`, only the first pasted image is attached, a large text becomes a chip
  const handlePaste = (e: React.ClipboardEvent) => {
    const videos = onVideo ? [...e.clipboardData.files].filter((file) => file.type.startsWith("video/")) : []
    if (videos.length > 0) {
      e.preventDefault()
      takeFiles(videos)
      return
    }
    const refused = accept ? [...e.clipboardData.files].filter((file) => !accept(file)) : []
    if (refused.length > 0) {
      e.preventDefault()
      onRefused?.(refused)
      return
    }
    for (const item of e.clipboardData.items) {
      if (item.type.startsWith("image/")) {
        e.preventDefault()
        const blob = item.getAsFile()
        if (!blob) continue
        upload(blob, blob.name || `pasted-image.${item.type.split("/")[1] || "png"}`)
        return
      }
    }
    const pasted = largePasteChip(e.clipboardData.getData("text/plain"))
    if (pasted) {
      e.preventDefault()
      chips.stage(pasted)
    }
  }

  // Only where videos are taken: dropped files join the draft like attached ones
  const handleDragOver = (e: React.DragEvent) => {
    if (onVideo && e.dataTransfer.types.includes("Files")) e.preventDefault()
  }
  const handleDrop = (e: React.DragEvent) => {
    if (!onVideo || e.dataTransfer.files.length === 0) return
    e.preventDefault()
    takeFiles([...e.dataTransfer.files])
  }

  const contextChip = ({ messageIndex, attachment }: ContextAttachment) => context && (
    <AttachmentChip
      key={`${messageIndex}-${attachment.name}`}
      attachment={attachment}
      onOpen={() => context.onOpen(messageIndex, attachment)}
      onRemove={() => context.onRemove(messageIndex, attachment.name)}
      onToggleActive={() => context.onToggleActive(messageIndex, attachment.name)}
    />
  )
  const activeContext = context?.attachments.filter(({ attachment }) => attachment.active) ?? []
  const inactiveContext = context?.attachments.filter(({ attachment }) => !attachment.active) ?? []
  const canSend = !picker && (text.trim().length > 0 || attachments.length > 0 || chips.chips.length > 0)
  const submitBlocked = Boolean(extracting) || (Boolean(isStreaming) && !sendWhileStreaming)
  const showStop = Boolean(isStreaming) && !(sendWhileStreaming && canSend)
  const hintText = hint ?? null
  const menuItems: (AttachMenuItem & { iconClassName?: string })[] = [
    ...attachMenuItems,
    { label: "Drawing", icon: Pencil, onClick: () => setDrawing("attach") },
    // Its output is a markdown file: offered only where the composer takes one
    ...(!accept || accept(new File([], "drawing-ocr.md", { type: "text/markdown" })) ? [{ label: "Drawing (OCR)", icon: Pencil, onClick: () => setDrawing("ocr"), iconClassName: "text-ink" }] : []),
  ]

  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} data-composer className="mx-auto w-full max-w-2xl">
      {((context?.attachments.length ?? 0) > 0 || attachments.length > 0 || chips.chips.length > 0) && (
        <div className="relative z-10 mb-2 flex flex-wrap gap-1.5">
          {activeContext.map(contextChip)}
          {attachments.map((attachment) => (
            <AttachmentChip
              key={attachment.name}
              attachment={attachment}
              uploading={drafts.uploadingNames.has(attachment.name)}
              onOpen={() => void drafts.openPreview(attachment)}
              onRemove={() => drafts.remove(attachment.name)}
            />
          ))}
          {chips.chips.map((chip) => <TextChip key={chip.id} chip={chip} onChange={(text) => chips.update(chip.id, text)} onRemove={() => chips.remove(chip.id)} />)}
          {inactiveContext.length > 0 && (
            <InactiveAttachments count={inactiveContext.length}>{inactiveContext.map(contextChip)}</InactiveAttachments>
          )}
        </div>
      )}

      <div onDragOver={handleDragOver} onDrop={handleDrop} className="relative z-20 rounded-2xl border border-divider/50 px-4 pb-2.5 pt-3.5 transition-all duration-300 bg-surface shadow-[var(--shadow-md),var(--inner-highlight)] focus-within:border-divider focus-within:shadow-[var(--shadow-md),var(--inner-highlight-strong)]">
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
            onChange={(e) => { setText(e.target.value); if (!picker) completions.request(e.target.value, e.target.selectionStart) }}
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
              {leading}
            </div>
            <div className="flex items-center gap-0.5">
              {toolsMenu && <ToolsMenu disabled={disabled} />}
              <button aria-label={isListening ? "Stop listening" : "Start listening"}
                onClick={toggleListening}
                disabled={disabled}
                className={cn(COMPOSER_ICON_BUTTON, isListening && "bg-danger-soft text-danger hover:bg-danger-soft hover:text-danger")}
              >
                <Mic className={cn("h-4 w-4", isListening && "animate-pulse")} />
              </button>
              <SendButton isStreaming={showStop} extracting={extracting} ready={canSend} disabled={disabled} onCancel={onCancel} onSend={handleSubmit} />
            </div>
          </div>
        </div>
      <DrawingAttach
        mode={drawing}
        chatId={chatId}
        onClose={() => setDrawing(null)}
        onAttach={drafts.add}
        onUpload={(file) => takeFiles([file])}
      />
      <AttachmentPreview attachment={drafts.preview} onClose={drafts.closePreview} onContentChange={drafts.editPreview} />
    </motion.div>
  )
})
