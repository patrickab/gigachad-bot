"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { ArrowUp, Plus, LayoutGrid, Mic, Search, Globe, Sigma, Square, X, Maximize2, FileText } from "lucide-react"
import { cn } from "@/lib/utils"
import { downscaleImage as apiDownscale } from "@/lib/api"
import { IMAGE_DOWNSCALE_MAX } from "@/lib/config"
import { useClickOutside } from "@/hooks/useClickOutside"
import { useModeState } from "@/hooks/useModeState"
import { useSettings } from "@/contexts/SettingsContext"
import { PillButton } from "./PillButton"

interface ChatInputProps {
  onSend: (text: string, imageDataUrl: string | null, pdfFiles?: File[] | null) => void
  onOCRRequest?: (image: string) => void
  disabled?: boolean
  isStreaming?: boolean
  onCancel?: () => void
}

export function ChatInput({
  onSend,
  onOCRRequest,
  disabled,
  isStreaming,
  onCancel,
}: ChatInputProps) {
  const { researchEnabled, morphicSearchEnabled, ocrEnabled, toggleResearch, toggleMorphicSearch, toggleOCR } = useModeState()
  const { downscaleImages } = useSettings()

  const [text, setText] = useState("")
  const [image, setImage] = useState<string | null>(null)
  const [downscaledImage, setDownscaledImage] = useState<string | null>(null)
  const [pdfFiles, setPdfFiles] = useState<File[]>([])
  const [showPreview, setShowPreview] = useState(false)
  const [isListening, setIsListening] = useState(false)
  const [showTools, setShowTools] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const toolsRef = useRef<HTMLDivElement>(null)
  const recognitionRef = useRef<any>(null)
  const textRef = useRef(text)

  const closeTools = useCallback(() => setShowTools(false), [])
  useClickOutside(toolsRef, closeTools)

  useEffect(() => { textRef.current = text }, [text])

  useEffect(() => {
    if (!image) { setDownscaledImage(null); return }
    apiDownscale(image, IMAGE_DOWNSCALE_MAX).then(setDownscaledImage).catch(() => {})
  }, [image])

  const adjustHeight = useCallback(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = "0"
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }, [])

  const clearImage = useCallback(() => {
    setImage(null)
    setDownscaledImage(null)
  }, [])

  const clearPdfs = useCallback(() => setPdfFiles([]), [])
  const clearPdf = useCallback((idx: number) => setPdfFiles(prev => prev.filter((_, i) => i !== idx)), [])

  const handleSubmit = useCallback(() => {
    const trimmed = text.trim()
    if (!trimmed && !image && pdfFiles.length === 0) return
    if (ocrEnabled && image) {
      onOCRRequest?.(image)
      setText("")
      clearImage()
      requestAnimationFrame(() => adjustHeight())
      return
    }
    onSend(trimmed || "", image, pdfFiles.length > 0 ? pdfFiles : null)
    setText("")
    clearImage()
    clearPdfs()
    requestAnimationFrame(() => adjustHeight())
  }, [text, image, pdfFiles, onSend, onOCRRequest, ocrEnabled, clearImage, clearPdfs, adjustHeight])

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
        const reader = new FileReader()
        reader.onload = () => setImage(reader.result as string)
        reader.readAsDataURL(blob)
        break
      }
    }
  }, [])

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files ?? [])
    if (selectedFiles.length === 0) return

    const pdfs: File[] = []
    let gotImage = false
    for (const file of selectedFiles) {
      if (file.type === "application/pdf") {
        pdfs.push(file)
      } else if (!gotImage) {
        const reader = new FileReader()
        reader.onload = () => setImage(reader.result as string)
        reader.readAsDataURL(file)
        gotImage = true
      }
    }
    if (pdfs.length > 0) {
      setPdfFiles(prev => [...prev, ...pdfs])
      if (!gotImage) { setImage(null); setDownscaledImage(null) }
    } else {
      setPdfFiles([])
    }
    e.target.value = ""
  }, [])

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

  const canSend = text.trim().length > 0 || !!image || pdfFiles.length > 0
  const previewSrc = (downscaleImages !== false && downscaledImage) ? downscaledImage : image
  const displaySrc = showPreview ? image : previewSrc

  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="mx-auto w-full max-w-3xl">
      <AnimatePresence>
        {showPreview && image && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-8"
            onClick={() => setShowPreview(false)}
          >
            <img
              src={image}
              alt="Full preview"
              className="max-h-full max-w-full rounded-lg object-contain"
            />
          </motion.div>
        )}
      </AnimatePresence>

      {image && (
        <motion.div
          initial={{ opacity: 0, scale: 0.96 }}
          animate={{ opacity: 1, scale: 1 }}
          className="relative mb-2"
        >
          <div className="group relative">
            <img
              src={previewSrc ?? image}
              alt="Attached"
              className="h-20 w-20 rounded-xl border border-zinc-700 object-cover cursor-pointer hover:border-zinc-500 transition-colors"
              onClick={() => setShowPreview(true)}
            />
            <button
              onClick={() => setShowPreview(true)}
              className="absolute right-1 bottom-1 rounded-md bg-black/60 p-0.5 text-zinc-400 opacity-0 group-hover:opacity-100 transition-opacity"
            >
              <Maximize2 className="h-3 w-3" />
            </button>
          </div>
          <button
            onClick={clearImage}
            className="absolute -right-1.5 -top-1.5 rounded-full bg-zinc-800 p-0.5 text-zinc-400 hover:text-red-400"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </motion.div>
      )}

      {pdfFiles.length > 0 && (
        <motion.div
          initial={{ opacity: 0, scale: 0.96 }}
          animate={{ opacity: 1, scale: 1 }}
          className="relative mb-2 flex flex-wrap gap-2"
        >
          {pdfFiles.map((file, idx) => (
            <motion.div key={`${file.name}-${idx}`} layout className="relative">
              <div className="flex items-center gap-1.5 rounded-xl border border-zinc-700 bg-zinc-800/40 px-2.5 py-1.5 pr-6">
                <FileText className="h-3.5 w-3.5 shrink-0 text-blue-400" />
                <span className="max-w-[160px] truncate text-xs text-zinc-300">{file.name}</span>
              </div>
              <button
                onClick={() => clearPdf(idx)}
                className="absolute -right-1 -top-1 rounded-full bg-zinc-800 p-0.5 text-zinc-400 hover:text-red-400"
              >
                <X className="h-3 w-3" />
              </button>
            </motion.div>
          ))}
        </motion.div>
      )}

      <div className="rounded-2xl border border-zinc-800/60 bg-zinc-900/90 shadow-xl shadow-black/30 backdrop-blur-sm p-4 transition-colors focus-within:border-zinc-700">
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
            <input ref={fileRef} type="file" accept="image/*,application/pdf" multiple onChange={handleFileSelect} className="hidden" />
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
                className="rounded-full p-2.5 transition-all bg-white text-black hover:bg-zinc-200"
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
                  canSend && !disabled ? "bg-white text-black hover:bg-zinc-200" : "bg-zinc-800 text-zinc-500"
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