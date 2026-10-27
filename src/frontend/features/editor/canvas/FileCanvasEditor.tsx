"use client"

import { useCallback, useEffect, useEffectEvent, useRef, useState } from "react"
import { motion } from "framer-motion"
import { CanvasEditor } from "./CanvasEditor"
import type { CanvasToolbarSlots } from "./CanvasToolbar"
import { emptyCanvasDoc, parseCanvasDoc, serializeCanvasDoc, type CanvasDocument } from "@/lib/domain/canvasDocument"
import { ApiError, loadFileViewerText } from "@/lib/backend/api"
import { subscribeToChanges } from "@/lib/backend/syncStream"
import { baseName } from "@/lib/paths"
import { useTabActive } from "@/contexts/TabContext"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import { LoadFailure, LoadingPlaceholder } from "@/components/ui/DocumentLoadState"

interface FileCanvasEditorProps {
  path: string
  slug: string
  onClose: () => void
  onSaved?: (filename?: string, content?: string) => void
  /** Writes the serialized canvas to its storage. */
  persistOverride: (content: string) => Promise<void>
  onModeLabel?: (label: string) => void
  canvasToolbarSlots?: CanvasToolbarSlots | null
}

/** Full-screen overlay for a file-root canvas: loads through the file viewer, autosaves through `persistOverride`. */
export function FileCanvasEditor({ path, slug, onClose, onSaved, persistOverride, onModeLabel, canvasToolbarSlots }: FileCanvasEditorProps) {
  const tabActive = useTabActive()
  const [canvasDoc, setCanvasDoc] = useState<CanvasDocument | null>(null)
  const [loadError, setLoadError] = useState(false)
  const dirtyRef = useRef(false)
  const savedContentRef = useRef("")

  const filename = baseName(path)

  const announceMode = useEffectEvent(() => onModeLabel?.(filename))
  useEffect(() => { announceMode() }, [filename])

  const loadFromServer = useCallback(() => {
    const apply = (text: string) => {
      setLoadError(false)
      const doc = text.trim() ? parseCanvasDoc(text) : emptyCanvasDoc()
      setCanvasDoc(doc)
      savedContentRef.current = serializeCanvasDoc(doc)
    }
    // A confirmed-missing document legitimately starts blank. Any other failure must
    // not arm a save of a blank canvas over the real stored content.
    return loadFileViewerText(path).then(apply).catch((err) => {
      if (err instanceof ApiError && err.status === 404) apply("")
      else setLoadError(true)
    })
  }, [path])

  useEffect(() => { loadFromServer() }, [loadFromServer])

  const handleChange = useCallback((doc: CanvasDocument) => {
    setCanvasDoc(doc)
    dirtyRef.current = true
  }, [])

  // Reload on remote changes unless there are unsaved local edits. `resource_key` may be a path suffix.
  useEffect(() => subscribeToChanges((event) => {
    if (dirtyRef.current || event.resource_kind !== "document") return
    if (event.resource_key !== path && !path.endsWith(`/${event.resource_key}`)) return
    loadFromServer()
  }), [path, loadFromServer])

  // Saves are chained so an earlier slow write can never resolve after,
  // and silently clobber, a newer one.
  const saveChain = useRef(Promise.resolve())
  const persist = useCallback((next: string) => {
    const run = saveChain.current.then(async () => {
      await persistOverride(next)
      savedContentRef.current = next
      dirtyRef.current = false
      onSaved?.(filename, next)
    })
    saveChain.current = run.catch(() => {})
    return run
  }, [persistOverride, filename, onSaved])

  // Autosave once the canvas has been quiet for a second.
  useEffect(() => {
    if (!canvasDoc) return
    const timer = setTimeout(() => {
      const serialized = serializeCanvasDoc(canvasDoc)
      if (serialized !== savedContentRef.current) persist(serialized).catch(() => {})
    }, 1000)
    return () => clearTimeout(timer)
  }, [canvasDoc, persist])

  // Leaving the editor flushes a pending autosave immediately. The debounce
  // cleanup alone would drop strokes drawn in the final second.
  const flush = useEffectEvent((force = false) => {
    if (!canvasDoc) return
    const serialized = serializeCanvasDoc(canvasDoc)
    if (force || serialized !== savedContentRef.current) persist(serialized).catch(() => {})
  })
  useEffect(() => () => flush(), [])

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && (event.key === "s" || event.key === "Enter")) {
        event.preventDefault()
        flush(true)
      }
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [])

  useEscapeLayer(onClose, tabActive)

  if (loadError) return <LoadFailure message="Failed to load this document. Its stored content was left untouched." onRetry={loadFromServer} />

  if (!canvasDoc) return <LoadingPlaceholder />

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.15 }}
      className="absolute inset-0 z-30 flex flex-row bg-paper"
    >
      <div className="flex-1 min-w-0 flex flex-col relative">
        <CanvasEditor doc={canvasDoc} onChange={handleChange} slug={slug} docPath={path} onImageAdded={() => onSaved?.()} toolbarSlots={canvasToolbarSlots} />
      </div>
    </motion.div>
  )
}
