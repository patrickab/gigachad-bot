"use client"

import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState, type ElementType } from "react"
import { createPortal } from "react-dom"
import { motion } from "framer-motion"
import { ArrowUp, Loader2, Plus, LayoutGrid, Mic, Square, Pencil } from "lucide-react"
import { blobToDataUrl, cn } from "@/lib/utils"
import { attachReference, uploadFile as apiUploadFile } from "@/lib/backend/api"
import { readReferences, saveAttachmentEdit, withReferenceRead } from "@/features/chat/attachments"
import type { Attachment } from "@/lib/domain/types"
import { useClickOutside } from "@/hooks/useClickOutside"
import { useEnabledTools } from "@/features/chat/EnabledToolsContext"
import { TOOLS } from "@/features/chat/tools"
import { useSettings } from "@/contexts/SettingsContext"
import { DrawingCanvas } from "./DrawingCanvas"
import { OCRPanel } from "@/features/editor/OCRPanel"
import { AttachmentPreview } from "@/features/chat/preview/AttachmentPreview"
import { AttachmentChip, type ContextAttachment } from "./AttachmentChip"
import { ToolMenuItem, ToolMenuSection } from "./ToolMenuItem"
import { CompletionMenu, useCompletions, type CompleteDraft } from "./CompletionMenu"

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
}

/** One recognition alternative; the browser API has no DOM lib typings. */
interface SpeechResultEvent {
  resultIndex: number
  results: ArrayLike<{ isFinal: boolean; 0: { transcript: string } }>
}

interface SpeechRecognitionLike {
  continuous: boolean
  interimResults: boolean
  lang: string
  onresult: ((event: SpeechResultEvent) => void) | null
  onend: (() => void) | null
  onerror: (() => void) | null
  start: () => void
  stop: () => void
}

type SpeechRecognitionConstructor = new () => SpeechRecognitionLike

const ICON_BUTTON = "rounded-full p-1.5 text-ink-faint transition-colors hover:bg-surface-elevated hover:text-ink focus-visible:outline focus-visible:outline-1 focus-visible:outline-ink-muted disabled:opacity-30"
/** Send and stop share one circle: raised surface when actionable, outlined when not. */
const SEND_BUTTON = "ml-1.5 flex h-8 w-8 items-center justify-center rounded-full border transition-all duration-200 focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-2 focus-visible:outline-ink-muted"
const SEND_READY = "border-divider bg-surface-elevated text-ink hover:border-divider-strong active:scale-[0.96]"
const SEND_IDLE = "border-divider/50 text-ink-faint"

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
  hint,
  context,
  attachMenuItems = [],
}, ref) {
  const { enabledTools, toggleTool } = useEnabledTools()
  const { ocrModel } = useSettings()

  const toolEntries = TOOLS.map((tool) => ({
    id: tool.name,
    label: tool.selectorLabel,
    icon: tool.icon,
    enabled: enabledTools.includes(tool.name),
    toggle: () => toggleTool(tool.name),
  }))

  const [text, setText] = useState("")
  const [attachments, setAttachments] = useState<Attachment[]>([])
  const [uploadingNames, setUploadingNames] = useState<Set<string>>(new Set())
  const [showTools, setShowTools] = useState(false)
  const [showAttachMenu, setShowAttachMenu] = useState(false)
  const [drawing, setDrawing] = useState<null | "attach" | "ocr">(null)
  const [drawingOcrImage, setDrawingOcrImage] = useState<string | null>(null)
  // Name of the draft attachment whose full preview is open. Earlier messages' chips open the chat's preview.
  const [previewName, setPreviewName] = useState<string | null>(null)
  const [isListening, setIsListening] = useState(false)

  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const toolsRef = useRef<HTMLDivElement>(null)
  const attachMenuRef = useRef<HTMLDivElement>(null)
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null)
  const textRef = useRef(text)

  const adjustHeight = useCallback(() => {
    const el = textareaRef.current
    if (!el) return
    // Reset first so scrollHeight can shrink
    el.style.height = "0"
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }, [])

  const upload = useCallback((file: File, name: string) => {
    setUploadingNames(prev => new Set(prev).add(name))
    apiUploadFile(chatId, file, slug)
      .then(att => setAttachments(prev => [...prev, att]))
      .catch(() => {})
      .finally(() => setUploadingNames(prev => { const n = new Set(prev); n.delete(name); return n }))
  }, [chatId, slug])
  const uploadFiles = useCallback((files: File[]) => {
    for (const file of files) upload(file, file.name)
  }, [upload])

  const replaceText = useCallback((next: string) => {
    setText(next)
    requestAnimationFrame(() => adjustHeight())
  }, [adjustHeight])
  const completions = useCompletions(complete, textareaRef, replaceText)

  useImperativeHandle(ref, () => ({
    addAttachment: (att: Attachment) => setAttachments(prev => {
      const existing = prev.findIndex(a => a.name === att.name)
      if (existing >= 0) { const copy = [...prev]; copy[existing] = att; return copy }
      return [...prev, att]
    }),
    updateText: (update) => {
      setText(update)
      requestAnimationFrame(() => { adjustHeight(); textareaRef.current?.focus() })
    },
    addFiles: uploadFiles,
  }), [adjustHeight, uploadFiles])

  const closeTools = useCallback(() => setShowTools(false), [])
  const closeAttachMenu = useCallback(() => setShowAttachMenu(false), [])
  useClickOutside(toolsRef, closeTools)
  useClickOutside(attachMenuRef, closeAttachMenu)

  // Mirror of `text` so speech callbacks read the latest value without re-binding
  useEffect(() => { textRef.current = text }, [text])

  const clearAttachments = useCallback(() => setAttachments([]), [])
  const removeAttachment = useCallback((name: string) => setAttachments(prev => prev.filter(a => a.name !== name)), [])
  const changeDraftContent = useCallback(async (name: string, content: string) => {
    const attachment = attachments.find((a) => a.name === name)
    if (!attachment) return
    const saved = await saveAttachmentEdit(attachment, content)
    setAttachments((prev) => prev.map((a) => a.name === name ? saved : a))
  }, [attachments])

  const handleSubmit = useCallback(() => {
    const trimmed = text.trim()
    if (!trimmed && attachments.length === 0) return
    completions.close()
    onSend(trimmed || "", attachments)
    setText("")
    clearAttachments()
    requestAnimationFrame(() => adjustHeight())
  }, [text, attachments, onSend, clearAttachments, adjustHeight, completions])

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (completions.handleKey(e, text)) return
    if (e.key === "Enter" && !e.shiftKey && !isStreaming && !extracting) { e.preventDefault(); handleSubmit() }
  }, [handleSubmit, isStreaming, extracting, completions, text])

  // Only the first pasted image is attached
  const handlePaste = useCallback((e: React.ClipboardEvent) => {
    const items = e.clipboardData.items
    for (const item of items) {
      if (item.type.startsWith("image/")) {
        e.preventDefault()
        const blob = item.getAsFile()
        if (!blob) continue
        upload(blob, blob.name || `pasted-image.${item.type.split("/")[1] || "png"}`)
        break
      }
    }
  }, [upload])

  // Stale `recognitionRef` checks ignore events from a recognizer that was already replaced or stopped
  const toggleListening = useCallback(() => {
    if (isListening) { recognitionRef.current?.stop(); setIsListening(false); return }
    const speechWindow = window as Window & { SpeechRecognition?: SpeechRecognitionConstructor; webkitSpeechRecognition?: SpeechRecognitionConstructor }
    const SpeechRecognition = speechWindow.SpeechRecognition || speechWindow.webkitSpeechRecognition
    if (!SpeechRecognition) { alert("Voice input is not supported."); return }
    const recognition = new SpeechRecognition()
    recognition.continuous = false
    recognition.interimResults = true
    recognition.lang = "en-US"
    let finalText = textRef.current.trim()
    let interimText = ""
    recognition.onresult = (event: SpeechResultEvent) => {
      let interim = ""
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const r = event.results[i]
        if (r.isFinal) finalText = finalText ? finalText + " " + r[0].transcript.trim() : r[0].transcript.trim()
        else interim += r[0].transcript
      }
      interimText = interim
      setText(finalText + (interimText ? " " + interimText : ""))
      requestAnimationFrame(() => adjustHeight())
    }
    recognition.onend = () => {
      if (interimText && recognitionRef.current === recognition) {
        finalText = finalText ? finalText + " " + interimText.trim() : interimText.trim()
        setText(finalText)
      }
      if (recognitionRef.current === recognition) setIsListening(false)
    }
    recognition.onerror = () => { if (recognitionRef.current === recognition) setIsListening(false) }
    recognitionRef.current = recognition
    recognition.start()
    setIsListening(true)
  }, [isListening, adjustHeight])

  const canSend = text.trim().length > 0 || attachments.length > 0
  const hintText = hint?.(text, attachments.length) ?? null
  // Resolved each render so the window follows edits and closes once its attachment is removed.
  const previewAttachment = attachments.find((a) => a.name === previewName) ?? null

  /** Open a draft attachment's preview, a reference re-read first so Neovim never edits a stale copy. */
  const openPreview = async (attachment: Attachment) => {
    if (attachment.vaultPath) {
      const reads = await readReferences([attachment.vaultPath], attachReference)
      setAttachments((prev) => prev.map((a) => a.name === attachment.name ? withReferenceRead(a, reads) : a))
    }
    setPreviewName(attachment.name)
  }

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
              uploading={uploadingNames.has(attachment.name)}
              onOpen={() => void openPreview(attachment)}
              onRemove={() => removeAttachment(attachment.name)}
            />
          ))}
        </div>
      )}

      <div className="relative z-20 rounded-2xl border border-divider/50 px-4 pb-2.5 pt-3.5 transition-all duration-300 bg-surface shadow-[var(--shadow-xl),var(--inner-highlight)] focus-within:border-divider focus-within:shadow-[var(--shadow-xl),var(--inner-highlight-strong)]">
          <CompletionMenu
            items={completions.items}
            activeIndex={completions.activeIndex}
            onHover={completions.setActiveIndex}
            onApply={(item) => completions.apply(text, item)}
          />
          <textarea
            ref={textareaRef}
            rows={1}
            value={text}
            onChange={(e) => { setText(e.target.value); adjustHeight(); completions.request(e.target.value, e.target.selectionStart) }}
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
              <div className="relative" ref={attachMenuRef}>
                <button aria-label="Add attachment"
                  onClick={() => setShowAttachMenu(!showAttachMenu)}
                  disabled={disabled}
                  className={cn(ICON_BUTTON, showAttachMenu && "bg-surface-elevated text-ink")}
                >
                  <Plus className="h-4 w-4" />
                </button>
                {showAttachMenu && (
                  <div className="absolute bottom-full left-0 mb-2 w-48 rounded-xl border border-divider bg-paper p-2 shadow-[var(--shadow-xl)]">
                    {menuItems.map(({ label, icon: Icon, onClick, iconClassName = "text-ink-muted" }) => (
                      <button
                        key={label}
                        onClick={() => { setShowAttachMenu(false); onClick() }}
                        className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-xs text-ink hover:bg-surface-elevated/50 transition-colors"
                      >
                        <Icon className={`h-3.5 w-3.5 ${iconClassName}`} />{label}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
            <div className="flex items-center gap-0.5">
              {toolsMenu && <div className="relative" ref={toolsRef}>
                <button aria-label="Toggle tools"
                  onClick={() => setShowTools(!showTools)}
                  disabled={disabled}
                  className={cn(ICON_BUTTON, showTools && "bg-surface-elevated text-ink")}
                >
                  <LayoutGrid className="h-4 w-4" />
                </button>
                {showTools && (
                  <div className="absolute bottom-full right-0 mb-2 w-56 rounded-xl border border-divider bg-paper p-2 shadow-[var(--shadow-xl)]">
                    <ToolMenuSection title="Tools">
                      {toolEntries.map(t => (
                        <ToolMenuItem key={t.id} icon={t.icon} label={t.label} active={t.enabled} onClick={t.toggle} />
                      ))}
                    </ToolMenuSection>
                  </div>
                )}
              </div>}
              <button aria-label={`${isListening ? "Stop listening" : "Start listening"}`}
                onClick={toggleListening}
                disabled={disabled}
                className={cn(ICON_BUTTON, isListening && "bg-danger-soft text-danger hover:bg-danger-soft hover:text-danger")}
              >
                <Mic className={cn("h-4 w-4", isListening && "animate-pulse")} />
              </button>
              {isStreaming ? (
                <button
                  onClick={onCancel}
                  className={cn(SEND_BUTTON, SEND_READY)}
                  aria-label="Stop generating"
                >
                  <Square className="h-3 w-3 fill-current" />
                </button>
              ) : extracting ? (
                <button aria-label="Extracting attachment" disabled className={cn(SEND_BUTTON, SEND_IDLE)}>
                  <Loader2 className="h-4 w-4 animate-spin" />
                </button>
              ) : (
                <button aria-label="Send message"
                  onClick={handleSubmit}
                  disabled={disabled || !canSend}
                  className={cn(SEND_BUTTON, canSend && !disabled ? SEND_READY : SEND_IDLE)}
                >
                  <ArrowUp className="h-4 w-4" />
                </button>
              )}
            </div>
          </div>
        </div>
      {drawing && (
        <DrawingCanvas
          chatId={chatId}
          onConfirm={(att) => {
            const mode = drawing
            setDrawing(null)

            // OCR drawings go to the OCR panel instead of being attached directly
            if (mode === "ocr") {
              fetch(att.url)
                .then(response => response.blob())
                .then(blobToDataUrl)
                .then(setDrawingOcrImage)
                .catch(() => {})
            } else {
              setAttachments(prev => [...prev, att])
            }
          }}
          onClose={() => setDrawing(null)}
        />
      )}
      {drawingOcrImage && createPortal(
        <div className="fixed inset-0 z-[90]">
          <OCRPanel
            image={drawingOcrImage}
            model={ocrModel}
            onComplete={async (output) => {
              setDrawingOcrImage(null)
              const blob = new Blob([output], { type: "text/markdown" })
              const file = new File([blob], `drawing-ocr-${Date.now()}.md`, { type: "text/markdown" })
              try {
                const att = await apiUploadFile(chatId, file, slug)
                setAttachments(prev => [...prev, att])
              } catch {}
            }}
            onClose={() => setDrawingOcrImage(null)}
          />
        </div>,
        document.body,
      )}
      <AttachmentPreview
        attachment={previewAttachment}
        onClose={() => setPreviewName(null)}
        onContentChange={previewName == null ? undefined : (newContent) => changeDraftContent(previewName, newContent)}
      />
    </motion.div>
  )
})
