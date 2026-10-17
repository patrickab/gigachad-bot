"use client"

import { useEffect } from "react"
import { createPortal } from "react-dom"
import { motion, AnimatePresence } from "framer-motion"
import { X } from "lucide-react"
import type { Attachment } from "@/lib/domain/types"
import { AttachmentView } from "@/features/chat/context-panel/AttachmentView"

interface AttachmentPreviewProps {
  attachment: Attachment | null
  onClose: () => void
}

/** Full-screen modal for an attachment. Renders nothing while `attachment` is null. */
export function AttachmentPreview({ attachment, onClose }: AttachmentPreviewProps) {
  useEffect(() => {
    if (!attachment) return
    function handler(e: KeyboardEvent) {
      if (e.key === "Escape") {
        e.preventDefault()
        onClose()
      }
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [attachment, onClose])

  if (!attachment) return null

  return createPortal(
    <AnimatePresence>
      <motion.div
        key="attachment-preview"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.15 }}
        className="fixed inset-0 z-[100] flex items-center justify-center bg-backdrop backdrop-blur-xl"
        onClick={onClose}
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.96, y: 8 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.96, y: 8 }}
          transition={{ duration: 0.2, ease: "easeOut" }}
          className="flex flex-col w-full max-w-4xl max-h-[85vh] mx-4 rounded-xl border border-divider-strong/50 bg-surface shadow-[var(--shadow-xl)] overflow-hidden"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="flex items-center justify-between px-4 py-2 border-b border-divider/50 shrink-0">
            <span className="text-xs font-medium text-ink truncate">{attachment.name}</span>
            <button aria-label="Close attachment preview"
              onClick={onClose}
              className="rounded p-1.5 text-ink-subtle hover:text-ink hover:bg-surface-elevated transition-colors"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
          <AttachmentView attachment={attachment} variant="modal" />
        </motion.div>
      </motion.div>
    </AnimatePresence>,
    document.body,
  )
}