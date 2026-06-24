"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { createPortal } from "react-dom"
import { Download, Maximize2, Minimize2, Save, X } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"
import { cn } from "@/lib/utils"
import { ConsoleEditor } from "./ConsoleEditor"
import { LaTeXMarkdown } from "./LaTeXMarkdown"
import { CanvasEditor, parseCanvasDoc, serializeCanvasDoc, emptyCanvasDoc, type CanvasDocument } from "./CanvasEditor"
import { loadFileViewerText, writeDocument, writeBinaryDocument, mirrorDrawing, fileViewerRawUrl } from "@/lib/api"
import { renderPageToPng, renderCanvasToJpeg, type EmbedRect } from "@/lib/drawing"

type EditorView = "edit" | "preview"

interface DocumentEditorProps {
  path: string
  slug: string
  onClose: () => void
  onSaved?: (filename?: string, content?: string) => void
  onLiveContent?: (path: string, content: string | null) => void
  availablePdfs?: { path: string; name: string }[]
  availableImages?: { path: string; name: string }[]
  overlay?: boolean
  persistOverride?: (content: string) => Promise<void>
  onModeLabel?: (label: string) => void
}

function editorLanguage(path: string): string {
  if (path.endsWith(".tex")) return "latex"
  return "markdown"
}

function ViewPills({ view, onViewChange }: { view: EditorView; onViewChange: (v: EditorView) => void }) {
  return (
    <div className="absolute top-2 left-3 z-10 flex items-center gap-px rounded-md bg-surface/80 backdrop-blur-sm p-0.5">
      {(["edit", "preview"] as const).map((v) => (
        <button
          key={v}
          onClick={() => onViewChange(v)}
          className={cn(
            "px-2 py-0.5 rounded text-[10px] font-medium transition-colors",
            view === v ? "bg-surface-elevated text-ink" : "text-ink-subtle hover:text-ink",
          )}
        >
          {v === "edit" ? "Edit" : "Preview"}
        </button>
      ))}
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

export function DocumentEditor({ path, slug, onClose, onSaved, onLiveContent, availablePdfs, availableImages, overlay, persistOverride, onModeLabel }: DocumentEditorProps) {
  const isCanvas = path.endsWith(".canvas")
  const [content, setContent] = useState<string | null>(null)
  const [canvasDoc, setCanvasDoc] = useState<CanvasDocument | null>(null)
  const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [isFullscreen, setIsFullscreen] = useState(!!overlay)
  const [view, setView] = useState<EditorView>("edit")
  const savedContentRef = useRef("")

  const filename = path.split("/").pop() ?? path
  const language = editorLanguage(path)

  // Overlay claims the tab label while mounted. Ref-stable to avoid re-render loops
  // (onModeLabel is an inline arrow that changes identity every render).
  const modeLabelRef = useRef(onModeLabel)
  useEffect(() => { modeLabelRef.current = onModeLabel }, [onModeLabel])
  useEffect(() => {
    if (!overlay) return
    modeLabelRef.current?.(filename)
  }, [overlay, filename])

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

  const buildImageEmbeds = useCallback(async (doc: CanvasDocument): Promise<EmbedRect[]> => {
    const images = doc.frames.filter((f) => f.kind === "image" && f.path)
    if (images.length === 0) return []
    return Promise.all(images.map(async (f) => {
      const url = fileViewerRawUrl(f.path!)
      let aspect = 1
      try {
        const img = new Image()
        img.crossOrigin = "anonymous"
        await new Promise<void>((resolve, reject) => { img.onload = () => resolve(); img.onerror = reject; img.src = url })
        if (img.naturalWidth > 0) aspect = img.naturalHeight / img.naturalWidth
      } catch { /* */ }
      return { url, x: f.x, y: f.y, width: f.width, aspect }
    }))
  }, [])

  const persist = useCallback(async (serialized: string) => {
    if (persistOverride) {
      await persistOverride(serialized)
      savedContentRef.current = serialized
      setDirty(false)
      onSaved?.(filename, serialized)
      return
    }
    await writeDocument(slug, filename, serialized)
    savedContentRef.current = serialized
    if (isCanvas && canvasDoc) savedCanvasKey.current = canvasContentKey(canvasDoc)
    setDirty(false)
    onSaved?.(filename, serialized)
    if (isCanvas && canvasDoc && (canvasDoc.strokes.length > 0 || canvasDoc.frames.some((f) => f.kind === "image"))) {
      try {
        const imgs = await buildImageEmbeds(canvasDoc)
        const blob = await renderCanvasToJpeg(canvasDoc.strokes, 20, imgs)
        await mirrorDrawing(filename.replace(/\.canvas$/, ".jpg"), blob)
      } catch { /* */ }
    }
  }, [slug, filename, isCanvas, canvasDoc, canvasContentKey, onSaved, buildImageEmbeds, persistOverride])

  const handleSave = useCallback(async () => {
    if (currentSerialized === null || saving) return
    setSaving(true)
    try { await persist(currentSerialized) } catch { /* */ }
    setSaving(false)
  }, [currentSerialized, saving, persist])

  const autoSaveTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  useEffect(() => {
    if (!isCanvas || !dirty || currentSerialized === null) return
    clearTimeout(autoSaveTimer.current)
    autoSaveTimer.current = setTimeout(() => { persist(currentSerialized).catch(() => {}) }, 1000)
    return () => clearTimeout(autoSaveTimer.current)
  }, [isCanvas, dirty, currentSerialized, persist])

  const handleExportPdf = useCallback(async () => {
    if (!isCanvas || !canvasDoc || exporting) return
    const pages = canvasDoc.frames.filter((f) => f.kind === "page")
    if (pages.length === 0) return
    setExporting(true)
    try {
      if (dirty && currentSerialized !== null) await persist(currentSerialized)
      const { PDFDocument } = await import("pdf-lib")
      const A4_ASPECT = 1123 / 794
      const imgs = await buildImageEmbeds(canvasDoc)
      const pdfDoc = await PDFDocument.create()
      for (const page of pages) {
        const pngBytes = await renderPageToPng(canvasDoc.strokes, page.x, page.y, page.width, page.width * A4_ASPECT, imgs)
        const img = await pdfDoc.embedPng(pngBytes)
        const pdfPage = pdfDoc.addPage([595.28, 841.89])
        pdfPage.drawImage(img, { x: 0, y: 0, width: 595.28, height: 841.89 })
      }
      const pdfBytes = await pdfDoc.save()
      const pdfName = filename.replace(/\.canvas$/, ".pdf")
      await writeBinaryDocument(slug, pdfName, new Blob([pdfBytes.buffer as ArrayBuffer], { type: "application/pdf" }))
      onSaved?.()
    } catch { /* */ }
    setExporting(false)
  }, [isCanvas, canvasDoc, exporting, dirty, currentSerialized, slug, filename, persist, onSaved, buildImageEmbeds])

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "s") {
        e.preventDefault()
        handleSave()
      }
      if (e.key === "Escape") {
        if (overlay) { e.preventDefault(); onClose() }
        else if (isFullscreen) { e.preventDefault(); setIsFullscreen(false) }
      }
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [handleSave, isFullscreen, overlay, onClose])

  const loaded = isCanvas ? canvasDoc !== null : content !== null
  if (!loaded) {
    return <div className="flex items-center justify-center py-6 text-xs text-ink-faint">Loading...</div>
  }

  const textBody = content !== null && !isCanvas && (
    <div className="relative flex-1 min-h-0 flex flex-col">
      <ViewPills view={view} onViewChange={setView} />
      {view === "edit" ? (
        <ConsoleEditor value={content} onChange={handleTextChange} language={language} placeholder="Start writing..." />
      ) : (
        <div className="flex-1 overflow-y-auto min-h-0 p-4 pt-10">
          <LaTeXMarkdown content={content} />
        </div>
      )}
    </div>
  )

  const canvasBody = isCanvas && canvasDoc && (
    <CanvasEditor doc={canvasDoc} onChange={handleCanvasChange} availablePdfs={availablePdfs} availableImages={availableImages} slug={slug} onImageAdded={() => onSaved?.()} />
  )

  const editorBody = textBody || canvasBody

  // Inline chrome (sidebar documents): filename, save, fullscreen, close.
  const chrome = !overlay && (
    <div className="flex items-center justify-between px-3 py-1.5 border-b border-divider/50 shrink-0">
      <div className="flex items-center gap-2 min-w-0">
        <span className="text-[11px] font-medium text-ink truncate">{filename}</span>
        {dirty && <span className="text-[10px] text-ink-faint">(modified)</span>}
      </div>
      <div className="flex items-center gap-1 shrink-0">
        <button onClick={handleSave} disabled={!dirty || saving} className="rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover disabled:opacity-30 transition-colors">
          <Save className="h-3.5 w-3.5" />
        </button>
        {isCanvas && (
          <button
            onClick={handleExportPdf}
            disabled={exporting || !canvasDoc || !canvasDoc.frames.some((f) => f.kind === "page")}
            className="rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover disabled:opacity-30 transition-colors"
          >
            {exporting
              ? <span className="h-3.5 w-3.5 block animate-spin rounded-full border-2 border-ink-faint border-t-ink" />
              : <Download className="h-3.5 w-3.5" />}
          </button>
        )}
        <button onClick={() => setIsFullscreen((f) => !f)} className="rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover transition-colors">
          {isFullscreen ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
        </button>
        <button onClick={onClose} className="rounded p-1 text-ink-subtle hover:text-danger transition-colors">
          <X className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  )

  if (overlay) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.15 }}
        className="absolute inset-0 z-30 flex flex-col bg-paper"
      >
        {editorBody}
      </motion.div>
    )
  }

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
