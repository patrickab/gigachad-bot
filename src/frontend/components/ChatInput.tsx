"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { motion } from "framer-motion"
import { ArrowUp, Plus, LayoutGrid, Mic, Search, Globe, X } from "lucide-react"
import { cn } from "@/lib/utils"

interface ChatInputProps {
  onSend: (text: string, imageDataUrl: string | null) => void
  disabled?: boolean
  researchEnabled?: boolean
  onResearchToggle?: () => void
  webSearchEnabled?: boolean
  onWebSearchToggle?: () => void
}

export function ChatInput({
  onSend,
  disabled,
  researchEnabled,
  onResearchToggle,
  webSearchEnabled,
  onWebSearchToggle,
}: ChatInputProps) {
  const [text, setText] = useState("")
  const [image, setImage] = useState<string | null>(null)
  const [isListening, setIsListening] = useState(false)
  const [showTools, setShowTools] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const toolsRef = useRef<HTMLDivElement>(null)
  const recognitionRef = useRef<any>(null)
  const textRef = useRef(text)

  useEffect(() => {
    textRef.current = text
  }, [text])

  useEffect(() => {
    function handler(e: MouseEvent) {
      if (toolsRef.current && !toolsRef.current.contains(e.target as Node)) setShowTools(false)
    }
    document.addEventListener("mousedown", handler)
    return () => document.removeEventListener("mousedown", handler)
  }, [])

  const adjustHeight = useCallback(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = "0"
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }, [])

  const handleSubmit = useCallback(() => {
    const trimmed = text.trim()
    if (!trimmed && !image) return
    onSend(trimmed || "Describe this image.", image)
    setText("")
    setImage(null)
    requestAnimationFrame(() => adjustHeight())
  }, [text, image, onSend, adjustHeight])

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault()
        handleSubmit()
      }
    },
    [handleSubmit]
  )

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
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => setImage(reader.result as string)
    reader.readAsDataURL(file)
    e.target.value = ""
  }, [])

  const toggleListening = useCallback(() => {
    if (isListening) {
      recognitionRef.current?.stop()
      setIsListening(false)
      return
    }

    const SpeechRecognition =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
    if (!SpeechRecognition) {
      alert("Voice input is not supported in your browser.")
      return
    }

    const recognition = new SpeechRecognition()
    recognition.continuous = false
    recognition.interimResults = true
    recognition.lang = "en-US"

    let finalText = textRef.current.trim()
    let interimText = ""

    recognition.onresult = (event: any) => {
      let interim = ""
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i]
        if (result.isFinal) {
          finalText = finalText
            ? finalText + " " + result[0].transcript.trim()
            : result[0].transcript.trim()
        } else {
          interim += result[0].transcript
        }
      }
      interimText = interim
      const display = finalText
        ? finalText + (interimText ? " " + interimText : "")
        : interimText
      setText(display)
      requestAnimationFrame(() => adjustHeight())
    }

    recognition.onend = () => {
      if (interimText && recognitionRef.current === recognition) {
        finalText = finalText
          ? finalText + " " + interimText.trim()
          : interimText.trim()
        setText(finalText)
      }
      if (recognitionRef.current === recognition) {
        setIsListening(false)
      }
    }

    recognition.onerror = () => {
      if (recognitionRef.current === recognition) {
        setIsListening(false)
      }
    }

    recognitionRef.current = recognition
    recognition.start()
    setIsListening(true)
  }, [isListening, adjustHeight])

  const canSend = text.trim().length > 0 || !!image

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="mx-auto w-full max-w-3xl"
    >
      {image && (
        <motion.div
          initial={{ opacity: 0, scale: 0.96 }}
          animate={{ opacity: 1, scale: 1 }}
          className="relative mb-2 inline-block"
        >
          <img src={image} alt="Attached" className="max-h-32 rounded-lg border border-zinc-800" />
          <button
            onClick={() => setImage(null)}
            className="absolute -right-1.5 -top-1.5 rounded-full bg-zinc-800 p-0.5 text-zinc-400 hover:text-red-400"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </motion.div>
      )}
      <div className="rounded-2xl border border-zinc-800/60 bg-zinc-900/90 shadow-xl shadow-black/30 backdrop-blur-sm p-4 transition-colors focus-within:border-zinc-700">
        <textarea
          ref={textareaRef}
          rows={1}
          value={text}
          onChange={(e) => {
            setText(e.target.value)
            adjustHeight()
          }}
          onKeyDown={handleKeyDown}
          onPaste={handlePaste}
          disabled={disabled}
          placeholder="Send a Message"
          className="w-full resize-none bg-transparent text-base text-zinc-100 placeholder:text-zinc-500 outline-none"
        />
        <div className="mt-3 flex items-center justify-between">
          <div className="flex items-center gap-1">
            <input ref={fileRef} type="file" accept="image/*" onChange={handleFileSelect} className="hidden" />
            <button
              onClick={() => fileRef.current?.click()}
              disabled={disabled}
              className="rounded-full p-2 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100 transition-colors disabled:opacity-30"
              title="Add resources"
            >
              <Plus className="h-4 w-4" />
            </button>
            {researchEnabled && (
              <button
                onClick={() => onResearchToggle?.()}
                className="flex items-center gap-1.5 rounded-full bg-amber-500/10 border border-amber-500/30 px-2.5 py-1 text-xs font-medium text-amber-400 hover:bg-amber-500/20 hover:border-amber-500/50 transition-colors"
                title="Deep Research — click to deactivate"
              >
                <Search className="h-3 w-3" />
                Research
              </button>
            )}
            {webSearchEnabled && (
              <button
                onClick={() => onWebSearchToggle?.()}
                className="flex items-center gap-1.5 rounded-full bg-sky-500/10 border border-sky-500/30 px-2.5 py-1 text-xs font-medium text-sky-400 hover:bg-sky-500/20 hover:border-sky-500/50 transition-colors"
                title="Web Search — click to deactivate"
              >
                <Globe className="h-3 w-3" />
                Search
              </button>
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
                      onClick={() => { onResearchToggle?.(); setShowTools(false) }}
                      className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-xs text-zinc-300 hover:bg-zinc-800/50 transition-colors"
                    >
                      <Search className="h-3.5 w-3.5 text-amber-400" />
                      Deep Research
                    </button>
                  )}
                  {!webSearchEnabled && (
                    <button
                      onClick={() => { onWebSearchToggle?.(); setShowTools(false) }}
                      className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-xs text-zinc-300 hover:bg-zinc-800/50 transition-colors"
                    >
                      <Globe className="h-3.5 w-3.5 text-sky-400" />
                      Web Search
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
                isListening
                  ? "bg-red-500/20 text-red-400 hover:bg-red-500/30"
                  : "text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100"
              )}
              title="Voice input"
            >
              <Mic className={cn("h-4 w-4", isListening && "animate-pulse")} />
            </button>
            <button
              onClick={handleSubmit}
              disabled={disabled || !canSend}
              className={cn(
                "rounded-full p-2.5 transition-all",
                canSend && !disabled
                  ? "bg-white text-black hover:bg-zinc-200"
                  : "bg-zinc-800 text-zinc-500"
              )}
            >
              <ArrowUp className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>
    </motion.div>
  )
}
