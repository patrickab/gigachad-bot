"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { createPortal } from "react-dom"
import { Download, Maximize2, Minimize2, Save, X } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"
import { cn } from "@/lib/utils"
import { ConsoleEditor } from "./ConsoleEditor"
import { LaTeXMarkdown } from "./LaTeXMarkdown"
import { CanvasEditor, parseCanvasDoc, serializeCanvasDoc, emptyCanvasDoc, type CanvasDocument } from "./CanvasEditor"
import { loadFileViewerText, writeDocument, writeBinaryDocument, mirrorDrawing } from "@/lib/api"
import { renderPageToPng, renderCanvasToJpeg } from "@/lib/drawing"

interface DocumentEditorProps {
  path: string
  slug: string
  onClose: () => void
  onSaved?: (filename?: string, content?: string) => void
  onLiveContent?: (path: string, content: string | null) => void
}

function editorLanguage(path: string): string {
  if (path.endsWith(".tex")) return "latex"
  return "markdown"
}

function TextEditor({ value, onChange, language, isFullscreen }: {
  value: string
  onChange: (v: string) => void
  language: string
  isFullscreen: boolean
}) {
  const [previewContent, setPreviewContent] = useState(value)

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault()
        setPreviewContent(value)
      }
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [value])

  if (!isFullscreen) {
    return (
      <div className="flex-1 min-h-0 flex flex-col">
        <ConsoleEditor value={value} onChange={onChange} language={language} placeholder="Start writing..." />
      </div>
    )
  }

  return (
    <div className="flex-1 min-h-0 flex">
      <div className="w-1/2 border-r border-divider/50 flex flex-col min-w-0">
        <div className="px-3 py-1.5 border-b border-divider/30 text-[10px] text-ink-faint font-medium uppercase tracking-wider">
          Editor
        </div>
        <ConsoleEditor value={value} onChange={onChange} language={language} placeholder="Start writing..." />
      </div>
      <div className="w-1/2 flex flex-col min-w-0">
        <div className="px-3 py-1.5 border-b border-divider/30 text-[10px] text-ink-faint font-medium uppercase tracking-wider flex items-center gap-2">
          Preview
          <span className="text-ink-faint">(Ctrl+Enter)</span>
        </div>
        <div className="flex-1 overflow-y-auto min-h-0 p-4">
          {previewContent ? (
            <LaTeXMarkdown content={previewContent} />
          ) : (
            <div className="text-xs text-ink-faint italic">Preview appears after Ctrl+Enter...</div>
          )}
        </div>
      </div>
    </div>
  )
}

function ResizableEditor({ children }: { children: React.ReactNode }) {
  const [height, setHeight] = useState(400)
  const dragging = useRef(false)
  const startY = useRef(0)
  const startH = useRef(0)

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    e.preventDefault()
    dragging.current = true
    startY.current = e.clientY
    startH.current = height
    ;(e.target as Element).setPointerCapture(e.pointerId)
  }, [height])

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    if (!dragging.current) return
    setHeight(Math.max(200, startH.current + (e.clientY - startY.current)))
  }, [])

  const onPointerUp = useCallback(() => { dragging.current = false }, [])

  return (
    <div className="flex flex-col" style={{ height }}>
      {children}
      <div
        className="h-1.5 cursor-row-resize shrink-0 flex items-center justify-center hover:bg-surface/60 transition-colors"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
      >
        <div className="w-8 h-0.5 rounded-full bg-divider-strong" />
      </div>
    </div>
  )
}

export function DocumentEditor({ path, slug, onClose, onSaved, onLiveContent }: DocumentEditorProps) {
  const isCanvas = path.endsWith(".canvas")
  const [content, setContent] = useState<string | null>(null)
  const [canvasDoc, setCanvasDoc] = useState<CanvasDocument | null>(null)
  const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const savedContentRef = useRef("")

  const filename = path.split("/").pop() ?? path
  const language = editorLanguage(path)

  useEffect(() => {
    loadFileViewerText(path).then((text) => {
      if (isCanvas) {
        const doc = text.trim() ? parseCanvasDoc(text) : emptyCanvasDoc()
        setCanvasDoc(doc)
        savedContentRef.current = serializeCanvasDoc(doc)
        savedCanvasKey.current = canvasContentKey(doc)
      } else {
        setContent(text)
        savedContentRef.current = text
      }
    }).catch(() => {
      if (isCanvas) {
        const doc = emptyCanvasDoc()
        setCanvasDoc(doc)
        savedContentRef.current = serializeCanvasDoc(doc)
        savedCanvasKey.current = canvasContentKey(doc)
      } else {
        setContent("")
      }
    })
  }, [path, isCanvas])

  const currentSerialized = isCanvas
    ? (canvasDoc ? serializeCanvasDoc(canvasDoc) : null)
    : content

  const handleTextChange = useCallback((v: string) => {
    setContent(v)
    setDirty(v !== savedContentRef.current)
  }, [])

  const canvasContentKey = useCallback((doc: CanvasDocument) => {
    const { viewport: _, ...rest } = doc
    return JSON.stringify(rest)
  }, [])

  const savedCanvasKey = useRef("")
  const handleCanvasChange = useCallback((doc: CanvasDocument) => {
    setCanvasDoc(doc)
    setDirty(canvasContentKey(doc) !== savedCanvasKey.current)
    onLiveContent?.(path, serializeCanvasDoc(doc))
  }, [canvasContentKey, onLiveContent, path])

  useEffect(() => {
    return () => { onLiveContent?.(path, null) }
  }, [path, onLiveContent])

  // single source of save bookkeeping — used by manual save, auto-save and export
  const persist = useCallback(async (serialized: string) => {
    await writeDocument(slug, filename, serialized)
    savedContentRef.current = serialized
    if (isCanvas && canvasDoc) savedCanvasKey.current = canvasContentKey(canvasDoc)
    setDirty(false)
    onSaved?.(filename, serialized)
    // mirror the rendered drawing into the cloud collection (md/tex mirror server-side).
    // ponytail: skip empty canvases so a cleared sketch leaves a stale .jpg rather than erroring
    if (isCanvas && canvasDoc?.strokes.length) {
      try {
        const blob = await renderCanvasToJpeg(canvasDoc.strokes)
        await mirrorDrawing(filename.replace(/\.canvas$/, ".jpg"), blob)
      } catch { /* */ }
    }
  }, [slug, filename, isCanvas, canvasDoc, canvasContentKey, onSaved])

  const handleSave = useCallback(async () => {
    if (currentSerialized === null || saving) return
    setSaving(true)
    try { await persist(currentSerialized) } catch { /* */ }
    setSaving(false)
  }, [currentSerialized, saving, persist])

  // auto-save canvas on content changes (debounced)
  const autoSaveTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  useEffect(() => {
    if (!isCanvas || !dirty || currentSerialized === null) return
    clearTimeout(autoSaveTimer.current)
    autoSaveTimer.current = setTimeout(() => { persist(currentSerialized).catch(() => {}) }, 1000)
    return () => clearTimeout(autoSaveTimer.current)
  }, [isCanvas, dirty, currentSerialized, persist])

  const handleExportPdf = useCallback(async () => {
    if (!isCanvas || !canvasDoc || exporting) return
    if (canvasDoc.pages.length === 0) return
    setExporting(true)
    try {
      if (dirty && currentSerialized !== null) await persist(currentSerialized)
      const { PDFDocument } = await import("pdf-lib")
      const A4_W = 794, A4_H = 1123
      const pdfDoc = await PDFDocument.create()
      for (const page of canvasDoc.pages) {
        const pngBytes = await renderPageToPng(canvasDoc.strokes, page.x, page.y, A4_W, A4_H)
        const img = await pdfDoc.embedPng(pngBytes)
        // A4 in PDF points: 595.28 x 841.89
        const pdfPage = pdfDoc.addPage([595.28, 841.89])
        pdfPage.drawImage(img, { x: 0, y: 0, width: 595.28, height: 841.89 })
      }
      const pdfBytes = await pdfDoc.save()
      const pdfName = filename.replace(/\.canvas$/, ".pdf")
      await writeBinaryDocument(slug, pdfName, new Blob([pdfBytes.buffer as ArrayBuffer], { type: "application/pdf" }))
      onSaved?.()
    } catch { /* */ }
    setExporting(false)
  }, [isCanvas, canvasDoc, exporting, dirty, currentSerialized, slug, filename, persist, onSaved])

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "s") {
        e.preventDefault()
        handleSave()
      }
      if (e.key === "Escape" && isFullscreen) {
        e.preventDefault()
        setIsFullscreen(false)
      }
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [handleSave, isFullscreen])

  const loaded = isCanvas ? canvasDoc !== null : content !== null
  if (!loaded) {
    return <div className="flex items-center justify-center py-6 text-xs text-ink-faint">Loading...</div>
  }

  const editorBody = isCanvas
    ? <CanvasEditor doc={canvasDoc!} onChange={handleCanvasChange} />
    : <TextEditor value={content!} onChange={handleTextChange} language={language} isFullscreen={isFullscreen} />

  const chrome = (
    <div className="flex items-center justify-between px-3 py-1.5 border-b border-divider/50 shrink-0">
      <div className="flex items-center gap-2 min-w-0">
        <span className="text-[11px] font-medium text-ink truncate">{filename}</span>
        {dirty && <span className="text-[10px] text-ink-faint">(modified)</span>}
      </div>
      <div className="flex items-center gap-1 shrink-0">
        <button
          onClick={handleSave}
          disabled={!dirty || saving}
          className="rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover disabled:opacity-30 transition-colors"
          title="Save (Ctrl+S)"
        >
          <Save className="h-3.5 w-3.5" />
        </button>
        {isCanvas && (
          <button
            onClick={handleExportPdf}
            disabled={exporting || !canvasDoc || canvasDoc.pages.length === 0}
            className="rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover disabled:opacity-30 transition-colors"
            title="Export pages to PDF"
          >
            {exporting
              ? <span className="h-3.5 w-3.5 block animate-spin rounded-full border-2 border-ink-faint border-t-ink" />
              : <Download className="h-3.5 w-3.5" />}
          </button>
        )}
        <button
          onClick={() => setIsFullscreen((f) => !f)}
          className="rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover transition-colors"
          title={isFullscreen ? "Exit fullscreen" : "Fullscreen"}
        >
          {isFullscreen ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
        </button>
        <button
          onClick={onClose}
          className="rounded p-1 text-ink-subtle hover:text-danger transition-colors"
          title="Close"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  )

  if (isFullscreen) {
    return createPortal(
      <AnimatePresence>
        <motion.div
          key="doc-editor-fullscreen"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
          className="fixed inset-0 z-[100] flex flex-col bg-paper"
        >
          {chrome}
          {editorBody}
        </motion.div>
      </AnimatePresence>,
      document.body,
    )
  }

  return (
    <ResizableEditor>
      {chrome}
      {editorBody}
    </ResizableEditor>
  )
}
