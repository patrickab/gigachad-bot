"use client"

import { useEffect } from "react"
import { createPortal } from "react-dom"
import { motion, AnimatePresence } from "framer-motion"
import { X } from "lucide-react"
import dynamic from "next/dynamic"
import { LaTeXMarkdown } from "@/components/LaTeXMarkdown"
import type { Attachment } from "@/lib/types"
import { rewriteImages } from "@/lib/api"

const PdfViewer = dynamic(() => import("./PdfViewer").then((m) => ({ default: m.PdfViewer })), { ssr: false })

function PreviewContent({ attachment, chatId, slug }: { attachment: Attachment; chatId: string; slug: string | null }) {
  if (attachment.mime.startsWith("image/")) {
    return (
      <div className="flex items-center justify-center p-6 overflow-auto max-h-[80vh]">
        <img src={attachment.url} alt={attachment.name} className="max-w-full max-h-[80vh] rounded-lg shadow-2xl object-contain" />
      </div>
    )
  }

  if (attachment.mime === "application/pdf") {
    return (
      <div className="flex-1 min-h-0 overflow-hidden">
        <PdfViewer url={attachment.url} />
      </div>
    )
  }

  const content = attachment.parsedMd ?? attachment.content
  if (content) {
    const rewritten = rewriteImages(content, chatId, slug)
    return (
      <div className="flex-1 min-h-0 overflow-y-auto p-6">
        <div className="max-w-3xl mx-auto">
          <LaTeXMarkdown content={rewritten} />
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 flex items-center justify-center">
      <p className="text-sm text-ink-subtle">No preview available</p>
    </div>
  )
}

interface AttachmentPreviewProps {
  attachment: Attachment | null
  chatId: string
  slug?: string | null
  onClose: () => void
}

export function AttachmentPreview({ attachment, chatId, slug = null, onClose }: AttachmentPreviewProps) {
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
          className="flex flex-col w-full max-w-4xl max-h-[85vh] mx-4 rounded-xl border border-divider-strong/50 bg-surface shadow-2xl overflow-hidden"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="flex items-center justify-between px-4 py-2 border-b border-divider/50 shrink-0">
            <span className="text-xs font-medium text-ink truncate">{attachment.name}</span>
            <button
              onClick={onClose}
              className="rounded p-1.5 text-ink-subtle hover:text-ink hover:bg-surface-elevated transition-colors"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
          <PreviewContent attachment={attachment} chatId={chatId} slug={slug} />
        </motion.div>
      </motion.div>
    </AnimatePresence>,
    document.body,
  )
}