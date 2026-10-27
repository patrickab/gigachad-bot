"use client"

import { useState } from "react"
import { createPortal } from "react-dom"
import { blobToDataUrl } from "@/lib/utils"
import { useSettings } from "@/contexts/SettingsContext"
import { OCRPanel } from "@/features/editor/OCRPanel"
import type { Attachment } from "@/lib/domain/types"
import { DrawingCanvas } from "./DrawingCanvas"

/** What a confirmed drawing becomes: the image itself, or the markdown the OCR panel reads from it. */
export type DrawingMode = "attach" | "ocr"

/**
 * The composer's drawing flow while `mode` is set, plus the OCR panel an "ocr" drawing opens.
 *
 * - "attach" hands the uploaded drawing to `onAttach`
 * - "ocr" opens the drawing in the OCR panel, whose confirmed output goes to `onUpload` as a markdown file
 */
export function DrawingAttach({ mode, chatId, onClose, onAttach, onUpload }: {
  mode: DrawingMode | null
  chatId: string
  onClose: () => void
  onAttach: (attachment: Attachment) => void
  onUpload: (file: File) => void
}) {
  const { ocrModel } = useSettings()
  const [ocrImage, setOcrImage] = useState<string | null>(null)
  return (
    <>
      {mode && (
        <DrawingCanvas
          chatId={chatId}
          onConfirm={(att) => {
            onClose()

            // OCR drawings go to the OCR panel instead of being attached directly
            if (mode === "ocr") {
              fetch(att.url)
                .then(response => response.blob())
                .then(blobToDataUrl)
                .then(setOcrImage)
                .catch(() => {})
            } else {
              onAttach(att)
            }
          }}
          onClose={onClose}
        />
      )}
      {ocrImage && createPortal(
        <div className="fixed inset-0 z-[90]">
          <OCRPanel
            image={ocrImage}
            model={ocrModel}
            onComplete={(output) => {
              setOcrImage(null)
              const blob = new Blob([output], { type: "text/markdown" })
              onUpload(new File([blob], `drawing-ocr-${Date.now()}.md`, { type: "text/markdown" }))
            }}
            onClose={() => setOcrImage(null)}
          />
        </div>,
        document.body,
      )}
    </>
  )
}
