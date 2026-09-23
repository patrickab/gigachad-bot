"use client"

import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from "react"
import { createPortal } from "react-dom"
import { motion, AnimatePresence } from "framer-motion"
import { ArrowUp, Loader2, Plus, LayoutGrid, Mic, Square, X, FileText, Image as ImageIcon, File as FileIcon, FileUp, Pencil } from "lucide-react"
import { cn } from "@/lib/utils"
import { ApiError, saveNotebook, uploadFile as apiUploadFile } from "@/lib/api"
import type { Attachment } from "@/lib/types"
import { useClickOutside } from "@/hooks/useClickOutside"
import { NOTEBOOK_TOOL_META, TOOLS, useModeState } from "@/hooks/useModeState"
import { useSettings } from "@/contexts/SettingsContext"
import { DrawingCanvas } from "./DrawingCanvas"
import { OCRPanel } from "./OCRPanel"
import { AttachmentPreview } from "./AttachmentPreview"
import { ToolMenuItem, ToolMenuSection } from "./ToolMenuItem"

interface ChatInputProps {
  chatId: string
  onSend: (text: string, attachments: Attachment[]) => void
  disabled?: boolean
  extracting?: boolean
  isStreaming?: boolean
  onCancel?: () => void
  slug?: string | null
}

export interface ChatInputHandle {
  addAttachment: (att: Attachment) => void
}

/** One markdown cell: exactly what activation writes to claim the first notebook revision. */
const STARTER_NOTEBOOK_SOURCE = "# %% [markdown]\n# Notebook\n"

function fileIcon(mime: string) {
  if (mime.startsWith("image/")) return ImageIcon
  if (mime === "application/pdf") return FileText
  return FileIcon
}

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
}, ref) {
  const { enabledTools, toggleTool, notebookChatId, setNotebookChatId } = useModeState()
  const { ocrModel } = useSettings()

  const toolEntries = TOOLS.map((tool) => ({
    id: tool.name,
    label: tool.selectorLabel,
    icon: tool.icon,
    enabled: enabledTools.includes(tool.name),
    toggle: () => toggleTool(tool.name),
  }))

  // Jupyter mode activation. The generation invalidates stale responses after a
  // chat switch, while the ref makes repeated clicks idempotent in one chat.
  const notebookStartingRef = useRef(false)
  const notebookGenerationRef = useRef(0)
  const notebookChatRef = useRef(chatId)
  if (notebookChatRef.current !== chatId) {
    notebookChatRef.current = chatId
    notebookGenerationRef.current += 1
    notebookStartingRef.current = false
  }
  const notebookActive = notebookChatId === chatId
  const activateNotebook = useCallback(() => {
    if (notebookActive || notebookStartingRef.current) return
    const generation = notebookGenerationRef.current
    notebookStartingRef.current = true
    saveNotebook(chatId, STARTER_NOTEBOOK_SOURCE, {}, "")
      .then(() => {
        if (generation === notebookGenerationRef.current) setNotebookChatId(chatId)
      })
      .catch((err) => {
        if (generation === notebookGenerationRef.current && err instanceof ApiError && err.status === 409) setNotebookChatId(chatId)
      })
      .finally(() => {
        if (generation === notebookGenerationRef.current) notebookStartingRef.current = false
      })
  }, [chatId, notebookActive, setNotebookChatId])

  const [text, setText] = useState("")
  const [attachments, setAttachments] = useState<Attachment[]>([])
  const [uploadingNames, setUploadingNames] = useState<Set<string>>(new Set())
  const [showTools, setShowTools] = useState(false)
  const [showAttachMenu, setShowAttachMenu] = useState(false)
  const [showDrawing, setShowDrawing] = useState(false)
  const [drawingOcrActive, setDrawingOcrActive] = useState(false)
  const [drawingOcrImage, setDrawingOcrImage] = useState<string | null>(null)
  const [previewAttachment, setPreviewAttachment] = useState<Attachment | null>(null)
  const [isListening, setIsListening] = useState(false)

  useImperativeHandle(ref, () => ({
    addAttachment: (att: Attachment) => setAttachments(prev => {
      const existing = prev.findIndex(a => a.name === att.name)
      if (existing >= 0) { const copy = [...prev]; copy[existing] = att; return copy }
      return [...prev, att]
    }),
  }), [])

  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const toolsRef = useRef<HTMLDivElement>(null)
  const attachMenuRef = useRef<HTMLDivElement>(null)
  const recognitionRef = useRef<any>(null)
  const textRef = useRef(text)

  const closeTools = useCallback(() => setShowTools(false), [])
  const closeAttachMenu = useCallback(() => setShowAttachMenu(false), [])
  useClickOutside(toolsRef, closeTools)
  useClickOutside(attachMenuRef, closeAttachMenu)

  useEffect(() => { textRef.current = text }, [text])

  const adjustHeight = useCallback(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = "0"
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }, [])

  const clearAttachments = useCallback(() => setAttachments([]), [])
  const removeAttachment = useCallback((name: string) => setAttachments(prev => prev.filter(a => a.name !== name)), [])

  const handleSubmit = useCallback(() => {
    const trimmed = text.trim()
    if (!trimmed && attachments.length === 0) return
    onSend(trimmed || "", attachments)
    setText("")
    clearAttachments()
    requestAnimationFrame(() => adjustHeight())
  }, [text, attachments, onSend, clearAttachments, adjustHeight])

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey && !isStreaming && !extracting) { e.preventDefault(); handleSubmit() }
  }, [handleSubmit, isStreaming])

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

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = Array.from(e.target.files ?? [])
    if (selected.length === 0) return
    for (const file of selected) {
      const name = file.name
      setUploadingNames(prev => new Set(prev).add(name))
      apiUploadFile(chatId, file, slug)
        .then(att => setAttachments(prev => [...prev, att]))
        .catch(() => {})
        .finally(() => setUploadingNames(prev => { const n = new Set(prev); n.delete(name); return n }))
    }
    e.target.value = ""
  }, [chatId, slug])

  const toggleListening = useCallback(() => {
    if (isListening) { recognitionRef.current?.stop(); setIsListening(false); return }
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
    if (!SpeechRecognition) { alert("Voice input is not supported."); return }
    const recognition = new SpeechRecognition()
    recognition.continuous = false
    recognition.interimResults = true
    recognition.lang = "en-US"
    let finalText = textRef.current.trim()
    let interimText = ""
    recognition.onresult = (event: any) => {
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

  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="mx-auto w-full max-w-2xl">
      {attachments.length > 0 && (
        <motion.div
          initial={{ opacity: 0, scale: 0.96 }}
          animate={{ opacity: 1, scale: 1 }}
          className="relative z-10 mb-2 flex flex-wrap gap-2"
        >
          {attachments.map((att) => {
            const Icon = fileIcon(att.mime)
            const uploading = uploadingNames.has(att.name)
            return (
              <motion.div key={att.name} layout className="relative w-44">
                <div
                  onClick={() => setPreviewAttachment(att)}
                  className="flex items-center gap-1.5 rounded-xl border border-divider-strong bg-surface-elevated px-2.5 py-1.5 pr-6 cursor-pointer hover:border-ink-muted transition-colors"
                >
                  {uploading ? (
                    <span className="h-3.5 w-3.5 shrink-0 animate-spin rounded-full border-2 border-ink-faint border-t-ink" />
                  ) : (
                    <Icon className="h-3.5 w-3.5 shrink-0 text-ink" />
                  )}
                  <span className="max-w-[160px] truncate text-xs text-ink">{att.name}</span>
                </div>
                <button
                  onClick={() => removeAttachment(att.name)}
                  className="absolute -right-1 -top-1 rounded-full bg-surface-elevated p-0.5 text-ink-muted hover:text-danger"
                >
                  <X className="h-3 w-3" />
                </button>
              </motion.div>
            )
          })}
        </motion.div>
      )}

      {/* Dark mode drops the composer one elevation step (--surface); light mode keeps
          --surface-elevated, where --surface sits too close to the canvas to read as a panel. */}
      <div className="relative z-20 rounded-2xl border border-divider-strong/60 p-4 transition-all duration-300 [background:color-mix(in_oklab,var(--surface-elevated),var(--ink)_4%)] dark:[background:color-mix(in_oklab,var(--surface),var(--ink)_4%)] shadow-[var(--shadow-lg),var(--inner-highlight)] focus-within:border-divider-strong focus-within:shadow-[var(--shadow-lg),var(--inner-highlight-strong)]">
          <textarea
            ref={textareaRef}
            rows={1}
            value={text}
            onChange={(e) => { setText(e.target.value); adjustHeight() }}
            onKeyDown={handleKeyDown}
            onPaste={handlePaste}
            disabled={disabled && !isStreaming}
            placeholder="Send a Message"
            spellCheck={false}
            autoCorrect="off"
            autoCapitalize="off"
            data-gramm="false"
            data-gramm_editor="false"
            data-enable-grammarly="false"
            className="w-full resize-none bg-transparent text-sm text-ink placeholder:text-ink-subtle outline-none"
          />
          <div className="mt-3 flex items-center justify-between">
            <div className="flex items-center gap-1">
              <input ref={fileRef} type="file" accept="image/*,application/pdf,.md,.txt,.csv,.json,.xml,.yaml,.yml,.toml,.rst,.log,.py,.js,.ts,.jsx,.tsx,.css,.html,.sh,.bash,.zsh,.go,.rs,.java,.c,.cpp,.h,.hpp,.rb,.php,.sql,.r,.tex,.bib" multiple onChange={handleFileSelect} className="hidden" />
              <div className="relative" ref={attachMenuRef}>
                <button
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
                    <button
                      onClick={() => { fileRef.current?.click(); setShowAttachMenu(false) }}
                      className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-xs text-ink hover:bg-surface-elevated/50 transition-colors"
                    >
                      <FileUp className="h-3.5 w-3.5 text-ink-muted" />Upload File
                    </button>
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
              <div className="relative" ref={toolsRef}>
                <button
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
                      <ToolMenuItem
                        icon={NOTEBOOK_TOOL_META.icon}
                        label={NOTEBOOK_TOOL_META.selectorLabel}
                        active={notebookActive}
                        onClick={activateNotebook}
                      />
                    </ToolMenuSection>

                  </div>
                )}
              </div>
              <button
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
                <button disabled className="rounded-full p-2.5 bg-surface-elevated text-ink-subtle">
                  <Loader2 className="h-4 w-4 animate-spin" />
                </button>
              ) : (
                <button
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
      <AttachmentPreview attachment={previewAttachment} chatId={chatId} slug={slug} onClose={() => setPreviewAttachment(null)} />
    </motion.div>
  )
})
