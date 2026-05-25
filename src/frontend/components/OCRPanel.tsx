"use client"

import { useEffect, useRef, useState, useCallback } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { X, Check } from "lucide-react"
import { MarkdownRenderer } from "./MarkdownRenderer"
import { ConsoleEditor } from "./ConsoleEditor"
import { createOCRStream } from "@/lib/api"

interface OCRPanelProps {
  image: string
  model: string
  onComplete: (output: string) => void
  onClose: () => void
}

export function OCRPanel({ image, model, onComplete, onClose }: OCRPanelProps) {
  const [output, setOutput] = useState("")
  const [isStreaming, setIsStreaming] = useState(true)
  const abortRef = useRef<(() => void) | null>(null)
  const confirmedRef = useRef(false)

  useEffect(() => {
    const { stream, abort } = createOCRStream(image, model)
    abortRef.current = abort
    let text = ""

    async function read() {
      try {
        while (true) {
          const { done, value } = await stream.read()
          if (done) break
          text += new TextDecoder().decode(value, { stream: true })
          setOutput(text)
        }
      } catch {
        // aborted or error
      } finally {
        setIsStreaming(false)
      }
    }
    read()

    return () => { abort() }
  }, [image, model])

  const doConfirm = useCallback(() => {
    if (confirmedRef.current) return
    confirmedRef.current = true
    abortRef.current?.()
    onComplete(output)
  }, [output, onComplete])

  useEffect(() => {
    function handler(e: KeyboardEvent) {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault()
        doConfirm()
      }
      if (e.key === "Escape") {
        onClose()
      }
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [doConfirm, onClose])

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="absolute inset-0 z-30 flex flex-col bg-zinc-950/95 backdrop-blur-sm"
      >
        <div className="flex items-center justify-between px-4 py-2 border-b border-zinc-800/50 shrink-0">
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="text-xs font-medium text-zinc-400">LaTeX OCR</span>
            {isStreaming && <span className="text-[10px] text-zinc-600">Streaming...</span>}
          </div>
          <div className="flex items-center gap-1 text-[10px] text-zinc-600">
            <kbd className="rounded border border-zinc-700 px-1.5 py-0.5 text-zinc-500">Ctrl+Enter</kbd>
            <span>to confirm</span>
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={doConfirm}
              disabled={!output}
              className="flex items-center gap-1.5 rounded-md bg-emerald-500/10 border border-emerald-500/30 px-3 py-1.5 text-xs font-medium text-emerald-400 hover:bg-emerald-500/20 disabled:opacity-30 transition-colors"
            >
              <Check className="h-3 w-3" />
              Confirm
            </button>
            <button
              onClick={onClose}
              className="p-1.5 text-zinc-500 hover:text-zinc-300 transition-colors"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        <div className="flex-1 flex min-h-0">
          <div className="w-1/2 border-r border-zinc-800/50 flex flex-col min-w-0">
            <div className="px-3 py-1.5 border-b border-zinc-800/30 text-[10px] text-zinc-600 font-medium uppercase tracking-wider">
              Console
            </div>
            <ConsoleEditor
              value={output}
              onChange={setOutput}
              language="latex"
              placeholder="Waiting for output..."
            />
          </div>
          <div className="w-1/2 flex flex-col min-w-0">
            <div className="px-3 py-1.5 border-b border-zinc-800/30 text-[10px] text-zinc-600 font-medium uppercase tracking-wider flex items-center gap-2">
              Preview
              <span className="text-zinc-700">(image → LaTeX)</span>
            </div>
            <div className="flex-1 overflow-y-auto min-h-0">
              <div className="p-4 flex flex-col">
                {output ? (
                  <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4 mb-3">
                    <MarkdownRenderer content={output} />
                  </div>
                ) : (
                  <div className="text-xs text-zinc-600 italic px-1 mb-3">Preview appears as output streams...</div>
                )}
                <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-1.5">
                  <img src={image} alt="Source" className="w-full rounded object-contain" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  )
}
