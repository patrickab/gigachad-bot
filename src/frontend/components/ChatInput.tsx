"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { ArrowUp, Plus, LayoutGrid, Mic, Search, Globe, Sigma, Square, X, FileText, Image as ImageIcon, File } from "lucide-react"
import { cn } from "@/lib/utils"
import { uploadFile as apiUploadFile } from "@/lib/api"
import type { Attachment } from "@/lib/types"
import { IMAGE_DOWNSCALE_MAX } from "@/lib/config"
import { downscaleImage as apiDownscale } from "@/lib/api"
import { useClickOutside } from "@/hooks/useClickOutside"
import { useModeState } from "@/hooks/useModeState"
import { useSettings } from "@/contexts/SettingsContext"
import { PillButton } from "./PillButton"

interface ChatInputProps {
  chatId: string
  onSend: (text: string, attachments: Attachment[]) => void
  onOCRRequest?: (imageDataUrl: string) => void
  disabled?: boolean
  isStreaming?: boolean
  onCancel?: () => void
}

function fileIcon(mime: string) {
  if (mime.startsWith("image/")) return ImageIcon
  if (mime === "application/pdf") return FileText
  return File
}

export function ChatInput({
  chatId,
  onSend,
  onOCRRequest,
  disabled,
  isStreaming,
  onCancel,
}: ChatInputProps) {
  const { researchEnabled, morphicSearchEnabled, ocrEnabled, toggleResearch, toggleMorphicSearch, toggleOCR } = useModeState()
  const { downscaleImages } = useSettings()

  const [text, setText] = useState("")
  const [attachments, setAttachments] = useState<Attachment[]>([])
  const [uploadingNames, setUploadingNames] = useState<Set<string>>(new Set())
  const [showTools, setShowTools] = useState(false)
  const [isListening, setIsListening] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const toolsRef = useRef<HTMLDivElement>(null)
  const recognitionRef = useRef<any>(null)
  const textRef = useRef(text)

  const closeTools = useCallback(() => setShowTools(false), [])
  useClickOutside(toolsRef, closeTools)

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
    if (ocrEnabled) {
      const imgAtt = attachments.find(a => a.mime.startsWith("image/"))
      if (imgAtt && onOCRRequest) {
        fetch(imgAtt.url)
          .then(r => r.blob())
          .then(blob => new Promise<string>((resolve) => {
            const reader = new FileReader()
            reader.onload = () => resolve(reader.result as string)
            reader.readAsDataURL(blob)
          }))
          .then(b64 => { onOCRRequest(b64); setText(""); clearAttachments(); requestAnimationFrame(() => adjustHeight()) })
          .catch(() => {})
        return
      }
    }
    onSend(trimmed || "", attachments)
    setText("")
    clearAttachments()
    requestAnimationFrame(() => adjustHeight())
  }, [text, attachments, onSend, onOCRRequest, ocrEnabled, clearAttachments, adjustHeight])

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSubmit() }
  }, [handleSubmit])

  const handlePaste = useCallback((e: React.ClipboardEvent) => {
    const items = e.clipboardData.items
    for (const item of items) {
      if (item.type.startsWith("image/")) {
        e.preventDefault()
        const blob = item.getAsFile()
        if (!blob) continue
        const name = blob.name || `pasted-image.${item.type.split("/")[1] || "png"}`
        setUploadingNames(prev => new Set(prev).add(name))
        apiUploadFile(chatId, blob)
          .then(att => setAttachments(prev => [...prev, att]))
          .catch(() => {})
          .finally(() => setUploadingNames(prev => { const n = new Set(prev); n.delete(name); return n }))
        break
      }
    }
  }, [chatId])

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = Array.from(e.target.files ?? [])
    if (selected.length === 0) return
    for (const file of selected) {
      const name = file.name
      setUploadingNames(prev => new Set(prev).add(name))
      apiUploadFile(chatId, file)
        .then(att => setAttachments(prev => [...prev, att]))
        .catch(() => {})
        .finally(() => setUploadingNames(prev => { const n = new Set(prev); n.delete(name); return n }))
    }
    e.target.value = ""
  }, [chatId])

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
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="mx-auto w-full max-w-3xl relative">
      <div className="absolute -top-8 left-0 right-0 h-8 bg-gradient-to-t from-zinc-950/80 to-transparent pointer-events-none" />
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
              <motion.div key={att.name} layout className="relative">
                <div className="flex items-center gap-1.5 rounded-xl border border-zinc-700 bg-zinc-800/40 px-2.5 py-1.5 pr-6">
                  {uploading ? (
                    <span className="h-3.5 w-3.5 shrink-0 animate-spin rounded-full border-2 border-zinc-600 border-t-zinc-300" />
                  ) : (
                    <Icon className="h-3.5 w-3.5 shrink-0 text-blue-400" />
                  )}
                  <span className="max-w-[160px] truncate text-xs text-zinc-300">{att.name}</span>
                </div>
                <button
                  onClick={() => removeAttachment(att.name)}
                  className="absolute -right-1 -top-1 rounded-full bg-zinc-800 p-0.5 text-zinc-400 hover:text-red-400"
                >
                  <X className="h-3 w-3" />
                </button>
              </motion.div>
            )
          })}
        </motion.div>
      )}

      <div className="relative z-10 rounded-2xl border border-zinc-700/40 bg-zinc-900/60 shadow-2xl shadow-black/50 backdrop-blur-xl p-4 transition-colors focus-within:border-zinc-600/50">
        <textarea
          ref={textareaRef}
          rows={1}
          value={text}
          onChange={(e) => { setText(e.target.value); adjustHeight() }}
          onKeyDown={handleKeyDown}
          onPaste={handlePaste}
          disabled={disabled}
          placeholder="Send a Message"
          className="w-full resize-none bg-transparent text-base text-zinc-100 placeholder:text-zinc-500 outline-none"
        />
        <div className="mt-3 flex items-center justify-between">
          <div className="flex items-center gap-1">
            <input ref={fileRef} type="file" accept="image/*,application/pdf,.md,.txt,.csv,.json,.xml,.yaml,.yml,.toml,.rst,.log,.py,.js,.ts,.jsx,.tsx,.css,.html,.sh,.bash,.zsh,.go,.rs,.java,.c,.cpp,.h,.hpp,.rb,.php,.sql,.r,.tex,.bib" multiple onChange={handleFileSelect} className="hidden" />
            <button
              onClick={() => fileRef.current?.click()}
              disabled={disabled}
              className="rounded-full p-2 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100 transition-colors disabled:opacity-30"
              title="Add resources"
            >
              <Plus className="h-4 w-4" />
            </button>
            {researchEnabled && (
              <PillButton accent="amber" active onClick={() => toggleResearch()} icon={<Search className="h-3 w-3" />}>
                Research
              </PillButton>
            )}
            {morphicSearchEnabled && (
              <PillButton accent="sky" active onClick={() => toggleMorphicSearch()} icon={<Globe className="h-3 w-3" />}>
                Search
              </PillButton>
            )}
            {ocrEnabled && (
              <PillButton accent="emerald" active onClick={() => toggleOCR()} icon={<Sigma className="h-3 w-3" />}>
                LaTeX
              </PillButton>
            )}
          </div>
          <div className="flex items-center gap-1">
            <div className="relative" ref={toolsRef}>
              <button
                onClick={() => setShowTools(!showTools)}
                disabled={disabled}
                className={cn(
                  "rounded-full p-2 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100 transition-colors disabled:opacity-30",
                  showTools && "bg-zinc-800 text-zinc-200"
                )}
                title="Tools"
              >
                <LayoutGrid className="h-4 w-4" />
              </button>
              {showTools && (
                <div className="absolute bottom-full right-0 mb-2 w-56 rounded-xl border border-zinc-800 bg-zinc-950 p-2 shadow-2xl">
                  {!researchEnabled && (
                    <button
                      onClick={() => { toggleResearch(); setShowTools(false) }}
                      className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-xs text-zinc-300 hover:bg-zinc-800/50 transition-colors"
                    >
                      <Search className="h-3.5 w-3.5 text-amber-400" />Deep Research
                    </button>
                  )}
                  {!morphicSearchEnabled && (
                    <button
                      onClick={() => { toggleMorphicSearch(); setShowTools(false) }}
                      className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-xs text-zinc-300 hover:bg-zinc-800/50 transition-colors"
                    >
                      <Globe className="h-3.5 w-3.5 text-sky-400" />Web Search
                    </button>
                  )}
                  {!ocrEnabled && (
                    <button
                      onClick={() => { toggleOCR(); setShowTools(false) }}
                      className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-xs text-zinc-300 hover:bg-zinc-800/50 transition-colors"
                    >
                      <Sigma className="h-3.5 w-3.5 text-emerald-400" />LaTeX OCR
                    </button>
                  )}
                </div>
              )}
            </div>
            <button
              onClick={toggleListening}
              disabled={disabled}
              className={cn(
                "rounded-full p-2 transition-colors disabled:opacity-30",
                isListening ? "bg-red-500/20 text-red-400 hover:bg-red-500/30" : "text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100"
              )}
              title="Voice input"
            >
              <Mic className={cn("h-4 w-4", isListening && "animate-pulse")} />
            </button>
            {isStreaming ? (
              <button
                onClick={onCancel}
                className="rounded-full p-2.5 transition-all bg-zinc-50 text-zinc-950 hover:bg-zinc-200"
                title="Stop generating"
              >
                <Square className="h-4 w-4 fill-current" />
              </button>
            ) : (
              <button
                onClick={handleSubmit}
                disabled={disabled || !canSend}
                className={cn(
                  "rounded-full p-2.5 transition-all",
                  canSend && !disabled ? "bg-zinc-50 text-zinc-950 hover:bg-zinc-200" : "bg-zinc-800 text-zinc-500"
                )}
              >
                <ArrowUp className="h-4 w-4" />
              </button>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  )
}
