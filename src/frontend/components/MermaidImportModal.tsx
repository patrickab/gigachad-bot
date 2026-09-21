"use client"

import { useEffect, useRef, useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Workflow } from "lucide-react"
import { cn } from "@/lib/utils"

interface MermaidImportModalProps {
  open: boolean
  onClose: () => void
  onImport: (source: string) => string | null
}

// Modeled on MindmapModal's prompt dialog, swapping the single-line input for
// a textarea and surfacing the parser's error inline instead of closing.
export function MermaidImportModal({ open, onClose, onImport }: MermaidImportModalProps) {
  const [source, setSource] = useState("")
  const [error, setError] = useState<string | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (open) {
      setSource("")
      setError(null)
      setTimeout(() => textareaRef.current?.focus(), 50)
    }
  }, [open])

  useEffect(() => {
    if (!open) return
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") onClose() }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [open, onClose])

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
          className="fixed inset-0 z-[100] flex items-center justify-center bg-backdrop backdrop-blur-[2px]"
          onClick={onClose}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.96, y: 8 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: 8 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            onClick={(e) => e.stopPropagation()}
            className="mx-4 w-full max-w-lg overflow-hidden rounded-xl border border-divider bg-paper shadow-[var(--shadow-xl)]"
          >
            <div className="px-5 pt-5 pb-3">
              <div className="flex items-center gap-2 text-sm font-medium text-ink">
                <Workflow className="h-4 w-4 text-ink" />
                Import from Mermaid
              </div>
              <p className="mt-1 text-xs text-ink-muted">Paste a flowchart. Node shapes, labels, and edges are preserved.</p>
            </div>
            <form
              onSubmit={(e) => {
                e.preventDefault()
                setError(onImport(source))
              }}
              className="px-5 pb-5"
            >
              <textarea
                ref={textareaRef}
                aria-label="Mermaid flowchart source"
                value={source}
                onChange={(e) => setSource(e.target.value)}
                placeholder={"flowchart TB\n    a[Start] --> b{Decide}"}
                rows={8}
                spellCheck={false}
                className={cn(
                  "w-full resize-none rounded-lg border border-divider bg-surface/60 px-3 py-2.5 font-mono text-xs text-ink placeholder-ink-faint",
                  "outline-none transition-all duration-200",
                  "focus:border-ink-muted",
                )}
              />
              {error && <p className="mt-2 text-xs text-danger" role="alert">{error}</p>}
              <div className="mt-3 flex justify-end gap-2">
                <button type="button" onClick={onClose} className="rounded-md px-3 py-1.5 text-xs text-ink-muted hover:bg-hover hover:text-ink">Cancel</button>
                <button type="submit" className="rounded-md bg-surface-elevated px-3 py-1.5 text-xs font-medium text-ink hover:bg-hover-strong">Import</button>
              </div>
            </form>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
