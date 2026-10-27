"use client"

import { useEffect } from "react"
import { motion } from "framer-motion"
import { CanvasEditor } from "./CanvasEditor"
import { emptyCanvasDoc, parseCanvasDoc, serializeCanvasDoc, type CanvasDocument } from "@/lib/domain/canvasDocument"
import { baseName } from "@/lib/paths"
import { useDisplayName } from "@/contexts/ProjectContext"
import { isModKey } from "@/lib/utils"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import { useEventListener } from "@/hooks/useEventListener"
import { useModeLabel } from "@/hooks/useModeLabel"
import { LoadFailure, LoadingPlaceholder } from "@/components/ui/DocumentLoadState"
import { useRootFileDocument, type DocumentCodec } from "../useRootFileDocument"
import { useDebouncedSave } from "../useDebouncedSave"

const AUTOSAVE_DELAY_MS = 1000

const CANVAS_CODEC: DocumentCodec<CanvasDocument> = {
  parse: (text) => text.trim() ? parseCanvasDoc(text) : emptyCanvasDoc(),
  serialize: serializeCanvasDoc,
}

interface FileCanvasEditorProps {
  path: string
  slug: string
  onClose: () => void
  onSaved?: () => void
  /** Writes the serialized canvas to its storage. */
  persistOverride: (content: string) => Promise<void>
  onModeLabel?: (label: string) => void
}

/** Full-screen overlay for a file-root canvas: loads through the file viewer, autosaves through `persistOverride`. */
export function FileCanvasEditor({ path, slug, onClose, onSaved, persistOverride, onModeLabel }: FileCanvasEditorProps) {
  // A confirmed-missing document legitimately starts blank.
  const { value: canvasDoc, loadError, reload, setValue, persist } = useRootFileDocument({
    path, codec: CANVAS_CODEC, write: persistOverride, blankWhenMissing: true, onSaved,
  })
  useEffect(() => { reload() }, [reload])
  const shownName = useDisplayName()(slug, path) ?? baseName(path)
  useModeLabel(shownName, true, onModeLabel)

  // Autosave once the canvas has been quiet for a second; a forced save also writes unchanged text.
  const flush = useDebouncedSave(canvasDoc, (doc, force) => { persist(doc, !force).catch(() => {}) }, AUTOSAVE_DELAY_MS)
  useEventListener("document", "keydown", (event) => {
    if (isModKey(event) && (event.key === "s" || event.key === "Enter")) {
      event.preventDefault()
      flush(true)
    }
  })
  useEscapeLayer(onClose)

  if (loadError) return <LoadFailure message="Failed to load this document. Its stored content was left untouched." onRetry={reload} />

  if (!canvasDoc) return <LoadingPlaceholder />

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.15 }}
      className="absolute inset-0 z-30 flex flex-row bg-paper"
    >
      <div className="flex-1 min-w-0 flex flex-col relative">
        <CanvasEditor doc={canvasDoc} onChange={setValue} slug={slug} docPath={path} onImageAdded={() => onSaved?.()} />
      </div>
    </motion.div>
  )
}
