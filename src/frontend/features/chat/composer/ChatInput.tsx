"use client"

import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState, type ElementType } from "react"
import { createPortal } from "react-dom"
import { motion } from "framer-motion"
import { ArrowUp, Loader2, Plus, LayoutGrid, Mic, Square, Pencil } from "lucide-react"
import { cn } from "@/lib/utils"
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

async function toDataUrl(url: string): Promise<string> {
  const blob = await (await fetch(url)).blob()
  return new Promise<string>((resolve) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as string)
    reader.readAsDataURL(blob)
  })
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
  placeholder = "Send a Message",
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
  const [showDrawing, setShowDrawing] = useState(false)
  const [drawingOcrActive, setDrawingOcrActive] = useState(false)
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

  const uploadFiles = useCallback((files: File[]) => {
    for (const file of files) {
      const name = file.name
      setUploadingNames(prev => new Set(prev).add(name))
      apiUploadFile(chatId, file, slug)
        .then(att => setAttachments(prev => [...prev, att]))
        .catch(() => {})
        .finally(() => setUploadingNames(prev => { const n = new Set(prev); n.delete(name); return n }))
    }
  }, [chatId, slug])

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
        const name = blob.name || `pasted-image.${item.type.split("/")[1] || "png"}`
        setUploadingNames(prev => new Set(prev).add(name))
        apiUploadFile(chatId, blob, slug)
          .then(att => setAttachments(prev => [...prev, att]))
          .catch(() => {})
          .finally(() => setUploadingNames(prev => { const n = new Set(prev); n.delete(name); return n }))
        break
      }
    }
  }, [chatId, slug])

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

      <div className="relative z-20 rounded-2xl border border-divider/50 p-4 transition-all duration-300 bg-surface shadow-[var(--shadow-xl),var(--inner-highlight)] focus-within:shadow-[var(--shadow-xl),var(--inner-highlight-strong)]">
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
            className="w-full resize-none bg-transparent text-sm text-ink placeholder:text-ink-subtle outline-none"
          />
          {hintText && <p className="mt-1 text-xs text-ink-muted">{hintText}</p>}
          <div className="mt-3 flex items-center justify-between">
            <div className="flex min-w-0 items-center gap-1">
              <div className="relative" ref={attachMenuRef}>
                <button aria-label="Add attachment"
                  onClick={() => setShowAttachMenu(!showAttachMenu)}
                  disabled={disabled}
                  className={cn(
                    "rounded-full p-2 text-ink-muted hover:bg-surface-elevated hover:text-ink transition-colors disabled:opacity-30",
                    showAttachMenu && "bg-surface-elevated text-ink"
                  )}

                >
                  <Plus className="h-4 w-4" />
                </button>
                {showAttachMenu && (
                  <div className="absolute bottom-full left-0 mb-2 w-48 rounded-xl border border-divider bg-paper p-2 shadow-[var(--shadow-xl)]">
                    {attachMenuItems.map(({ label, icon: Icon, onClick }) => (
                      <button
                        key={label}
                        onClick={() => { setShowAttachMenu(false); onClick() }}
                        className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-xs text-ink hover:bg-surface-elevated/50 transition-colors"
                      >
                        <Icon className="h-3.5 w-3.5 text-ink-muted" />{label}
                      </button>
                    ))}
                    <button
                      onClick={() => { setShowAttachMenu(false); setShowDrawing(true) }}
                      className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-xs text-ink hover:bg-surface-elevated/50 transition-colors"
                    >
                      <Pencil className="h-3.5 w-3.5 text-ink-muted" />Drawing
                    </button>
                    <button
                      onClick={() => { setShowAttachMenu(false); setDrawingOcrActive(true); setShowDrawing(true) }}
                      className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-xs text-ink hover:bg-surface-elevated/50 transition-colors"
                    >
                      <Pencil className="h-3.5 w-3.5 text-ink" />Drawing (OCR)
                    </button>
                  </div>
                )}
              </div>
            </div>
            <div className="flex items-center gap-1">
              {toolsMenu && <div className="relative" ref={toolsRef}>
                <button aria-label="Toggle tools"
                  onClick={() => setShowTools(!showTools)}
                  disabled={disabled}
                  className={cn(
                    "rounded-full p-2 text-ink-muted hover:bg-surface-elevated hover:text-ink transition-colors disabled:opacity-30",
                    showTools && "bg-surface-elevated text-ink"
                  )}

                >
                  <LayoutGrid className="h-4 w-4" />
                </button>
                {showTools && (
                  <div className="absolute bottom-full right-0 mb-2 w-56 rounded-xl border border-divider bg-paper p-2 shadow-[var(--shadow-xl)]">
                    <ToolMenuSection title="Tools" first>
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
                className={cn(
                  "rounded-full p-2 transition-colors disabled:opacity-30",
                  isListening ? "bg-danger-soft text-danger hover:bg-danger-soft" : "text-ink-muted hover:bg-surface-elevated hover:text-ink"
                )}

              >
                <Mic className={cn("h-4 w-4", isListening && "animate-pulse")} />
              </button>
              {isStreaming ? (
                <button
                  onClick={onCancel}
                  className="rounded-full p-2.5 transition-all duration-300 bg-ink text-paper hover:opacity-90 active:scale-[0.96]"
                  aria-label="Stop generating"
                >
                  <Square className="h-4 w-4 fill-current" />
                </button>
              ) : extracting ? (
                <button aria-label="Extracting attachment" disabled className="rounded-full p-2.5 bg-surface-elevated text-ink-subtle">
                  <Loader2 className="h-4 w-4 animate-spin" />
                </button>
              ) : (
                <button aria-label="Send message"
                  onClick={handleSubmit}
                  disabled={disabled || !canSend}
                  className={cn(
                    "rounded-full p-2.5 transition-all duration-300",
                    canSend && !disabled ? "bg-ink text-paper hover:opacity-90 active:scale-[0.96]" : "bg-surface-elevated text-ink-subtle"
                  )}
                >
                  <ArrowUp className="h-4 w-4" />
                </button>
              )}
            </div>
          </div>
        </div>
      {showDrawing && (
        <DrawingCanvas
          chatId={chatId}
          onConfirm={(att) => {
            setShowDrawing(false)

            // OCR drawings go to the OCR panel instead of being attached directly
            if (drawingOcrActive) {
              setDrawingOcrActive(false)
              toDataUrl(att.url)
                .then(b64 => setDrawingOcrImage(b64))
                .catch(() => {})
            } else {
              setAttachments(prev => [...prev, att])
            }
          }}
          onClose={() => { setShowDrawing(false); setDrawingOcrActive(false) }}
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
