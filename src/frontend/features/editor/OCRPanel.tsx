"use client"

import { useEffect, useRef, useState, useCallback } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { X, Check, ChevronLeft, ChevronRight } from "lucide-react"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"
import { AssistedConsoleEditor } from "./AssistedConsoleEditor"
import { EditorSidebar } from "./EditorSidebar"
import { createOCRStream } from "@/lib/backend/api"
import { useEventListener } from "@/hooks/useEventListener"
import { useResetOnChange } from "@/hooks/useResetOnChange"
import { blobToDataUrl } from "@/lib/utils"

interface OCRPanelProps {
  image: string
  model: string
  onComplete: (output: string) => void
  onClose: () => void
}

export function OCRPanel({ image, model, onComplete, onClose }: OCRPanelProps) {
  // The image lives in state (seeded from the prop) so the panel can be reset
  // in place: Alt+R clears everything and waits for a newly pasted image.
  const [img, setImg] = useState<string | null>(image)
  const [output, setOutput] = useState("")
  const [isStreaming, setIsStreaming] = useState(true)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [sidebarWidth, setSidebarWidth] = useState(340)
  const abortRef = useRef<(() => void) | null>(null)
  // A new image or model restarts OCR, so the stream is live again (the effect below starts it)
  useResetOnChange(img, (next) => { if (next) setIsStreaming(true) })
  useResetOnChange(model, () => { if (img) setIsStreaming(true) })
  // Set once confirmed so a second confirm cannot call onComplete again
  const confirmedRef = useRef(false)

  const reset = useCallback(() => {
    abortRef.current?.()
    setImg(null)
    setOutput("")
    setIsStreaming(false)
  }, [])

  // While awaiting a new image, a pasted image restarts OCR
  useEventListener("window", "paste", (e) => {
    const item = [...(e.clipboardData?.items ?? [])].find((i) => i.type.startsWith("image/"))
    const blob = item?.getAsFile()
    if (!blob) return
    e.preventDefault()
    blobToDataUrl(blob).then(setImg).catch(() => {})
  }, { active: !img })

  // Restart OCR whenever the image or model changes and abort the stream on cleanup
  useEffect(() => {
    if (!img) return
    const stream = createOCRStream(img, model)
    abortRef.current = stream.abort
    let text = ""

    async function read() {
      try {
        for await (const event of stream) {
          if (event.event === "token") {
            text += event.data
            setOutput(text)
          } else if (event.event === "done") {
            break
          } else if (event.event === "error") {
            text += `\n\nError: ${event.data}`
            setOutput(text)
            break
          }
        }
      } catch {
        // aborted or error
      } finally {
        setIsStreaming(false)
      }
    }
    read()

    return () => { stream.abort() }
  }, [img, model])

  // Guard against double confirm (key repeat, click plus shortcut)
  const doConfirm = useCallback(() => {
    if (confirmedRef.current) return
    confirmedRef.current = true
    abortRef.current?.()
    onComplete(output)
  }, [output, onComplete])

  useEventListener("document", "keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault()
      doConfirm()
    }
    if (e.altKey && e.key.toLowerCase() === "r") {
      e.preventDefault()
      reset()
    }
    if (e.key === "Escape") {
      onClose()
    }
  })

  const handleSidebarApply = useCallback((newContent: string) => {
    setOutput(newContent)
  }, [])

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="absolute inset-0 z-30 flex flex-col bg-paper/95 backdrop-blur-sm"
      >
        <div className="flex items-center justify-between px-4 py-2 border-b border-divider/50 shrink-0">
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-ink animate-pulse" />
            <span className="text-xs font-medium text-ink-muted">LaTeX OCR</span>
            {isStreaming && <span className="text-[10px] text-ink-faint">Streaming...</span>}
          </div>
          <div className="flex items-center gap-1 text-[10px] text-ink-faint">
            <kbd className="rounded border border-divider-strong px-1.5 py-0.5 text-ink-subtle">Ctrl+Enter</kbd>
            <span>confirm</span>
            <span className="mx-0.5">·</span>
            <kbd className="rounded border border-divider-strong px-1.5 py-0.5 text-ink-subtle">Alt+R</kbd>
            <span>new image</span>
          </div>
          <div className="flex items-center gap-1">
            <button
              aria-label={sidebarOpen ? "Hide OCR sidebar" : "Show OCR sidebar"}
              onClick={() => setSidebarOpen((v) => !v)}
              className="rounded p-1.5 text-ink-subtle hover:text-ink hover:bg-hover transition-colors"
            >
              {sidebarOpen ? <ChevronRight className="h-3.5 w-3.5" /> : <ChevronLeft className="h-3.5 w-3.5" />}
            </button>
            <button
              onClick={doConfirm}
              disabled={!output}
              className="flex items-center gap-1.5 rounded-md bg-surface-elevated border border-divider-strong px-3 py-1.5 text-xs font-medium text-ink hover:bg-surface-elevated disabled:opacity-30 transition-colors"
            >
              <Check className="h-3 w-3" />
              Confirm
            </button>
            <button aria-label="Close OCR panel"
              onClick={onClose}
              className="p-1.5 text-ink-subtle hover:text-ink transition-colors"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        <div className="flex-1 flex min-h-0">
          <div className="flex-1 min-w-0 flex min-h-0">
            <div className="w-1/2 border-r border-divider/50 flex flex-col min-w-0">
              <div className="px-3 py-1.5 border-b border-divider/30 text-[10px] text-ink-faint font-medium uppercase tracking-wider">
                Console
              </div>
              {/* Remounting when the image is cleared (Alt+R) drops any open inline edit */}
              <AssistedConsoleEditor key={img === null ? "awaiting-image" : "image"} value={output} onChange={setOutput} language="latex" model={model} />
            </div>
            <div className="w-1/2 flex flex-col min-w-0">
              <div className="px-3 py-1.5 border-b border-divider/30 text-[10px] text-ink-faint font-medium uppercase tracking-wider flex items-center gap-2">
                Preview
                <span className="text-ink-faint">(image → LaTeX)</span>
              </div>
              <div className="flex-1 overflow-y-auto min-h-0">
                <div className="p-4 flex flex-col">
                  {output ? (
                    <div className="rounded-lg border border-divider bg-surface/50 p-4 mb-3">
                      <LaTeXMarkdown content={output} />
                    </div>
                  ) : (
                    <div className="text-xs text-ink-faint italic px-1 mb-3">Preview appears as output streams...</div>
                  )}
                  {img ? (
                    <div className="rounded-lg border border-divider bg-surface/50 p-1.5">
                      {/* eslint-disable-next-line @next/next/no-img-element -- user-supplied pasted/blob data URL, next/image cannot optimize it */}
                      <img src={img} alt="Source" className="w-full rounded object-contain" />
                    </div>
                  ) : (
                    <div className="rounded-lg border border-dashed border-divider-strong bg-surface/30 px-4 py-10 flex flex-col items-center gap-1 text-center">
                      <span className="text-xs font-medium text-ink-muted">Paste a new image</span>
                      <span className="text-[10px] text-ink-faint">Ctrl+V — OCR starts automatically</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
          {sidebarOpen && (
            <EditorSidebar
              content={output}
              model={model}
              width={sidebarWidth}
              onWidthChange={setSidebarWidth}
              onApply={handleSidebarApply}
            />
          )}
        </div>
      </motion.div>
    </AnimatePresence>
  )
}
