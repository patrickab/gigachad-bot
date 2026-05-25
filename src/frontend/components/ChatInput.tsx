"use client"

import { useCallback, useRef, useState } from "react"
import { motion } from "framer-motion"
import { ArrowUp, ImagePlus, X } from "lucide-react"

interface ChatInputProps {
  onSend: (text: string, imageDataUrl: string | null) => void
  disabled?: boolean
}

export function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [text, setText] = useState("")
  const [image, setImage] = useState<string | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)

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
      <div className="flex items-end gap-2 rounded-xl border border-zinc-800 bg-zinc-900/80 px-3 py-2 transition-colors focus-within:border-zinc-700">
        <input ref={fileRef} type="file" accept="image/*" onChange={handleFileSelect} className="hidden" />
        <button
          onClick={() => fileRef.current?.click()}
          className="shrink-0 rounded-lg p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300 transition-colors"
          title="Attach image"
        >
          <ImagePlus className="h-4 w-4" />
        </button>
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
          placeholder="Send a message…"
          className="flex-1 resize-none bg-transparent text-sm text-zinc-100 placeholder:text-zinc-500 outline-none"
        />
        <button
          onClick={handleSubmit}
          disabled={disabled || (!text.trim() && !image)}
          className="shrink-0 rounded-lg p-1.5 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-100 disabled:opacity-30 transition-colors"
        >
          <ArrowUp className="h-4 w-4" />
        </button>
      </div>
    </motion.div>
  )
}
