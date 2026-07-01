"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { createPortal } from "react-dom"
import { ChevronLeft, ChevronRight, Download, Maximize2, Minimize2, Save, X } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"
import { cn } from "@/lib/utils"
import { ConsoleEditor } from "./ConsoleEditor"
import { LaTeXMarkdown } from "./LaTeXMarkdown"
import { CanvasEditor, parseCanvasDoc, serializeCanvasDoc, emptyCanvasDoc, type CanvasDocument } from "./CanvasEditor"
import { loadFileViewerText, readFileVaultRendered, writeDocument, writeBinaryDocument, mirrorDrawing, fileViewerRawUrl } from "@/lib/api"
import { renderPageToPng, renderCanvasToJpeg, type EmbedRect } from "@/lib/drawing"
import { EditorSidebar, InlineEditPanel } from "./EditorSidebar"

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
  onNavigate?: (path: string) => void
  model?: string
}

function editorLanguage(path: string): string {
  if (path.endsWith(".tex")) return "latex"
  return "markdown"
}

function ViewPills({ view, onViewChange }: { view: EditorView; onViewChange: (v: EditorView) => void }) {
  return (
    <div className="absolute top-2 left-3 z-10 flex items-center gap-px rounded-md bg-surface/80 backdrop-blur-sm p-0.5 opacity-0 hover:opacity-100 transition-opacity">
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

export function DocumentEditor({ path, slug, onClose, onSaved, onLiveContent, availablePdfs, availableImages, overlay, persistOverride, onModeLabel, onNavigate, model }: DocumentEditorProps) {
  const isCanvas = path.endsWith(".canvas")
  const [content, setContent] = useState<string | null>(null)
  const [renderedContent, setRenderedContent] = useState<string | null>(null)
  const [canvasDoc, setCanvasDoc] = useState<CanvasDocument | null>(null)
  const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false)
  const [exporting, setExporting] = useState(false)
  const [isFullscreen, setIsFullscreen] = useState(!!overlay)
  const [view, setView] = useState<EditorView>("edit")
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [sidebarWidth, setSidebarWidth] = useState(340)
  const [inlineEdit, setInlineEdit] = useState<{ text: string; start: number; end: number; splitPx: number } | null>(null)
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
    setRenderedContent(null)
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
    if (overlay && !isCanvas) {
      readFileVaultRendered(path).then(setRenderedContent).catch(() => {})
    }
  }, [path, isCanvas, overlay])

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

  const handlePreviewClick = useCallback((e: React.MouseEvent) => {
    const anchor = (e.target as HTMLElement).closest("a")
    if (!anchor) return
    const href = anchor.getAttribute("href")
    if (!href?.startsWith("#vault:")) return
    e.preventDefault()
    onNavigate?.(decodeURIComponent(href.slice("#vault:".length)))
  }, [onNavigate])

  const handleSidebarApply = useCallback((newContent: string) => {
    setContent(newContent)
    setDirty(newContent !== savedContentRef.current)
  }, [])

  const handleInlineApply = useCallback((replacement: string) => {
    if (!inlineEdit || content === null) return
    const next = content.substring(0, inlineEdit.start) + replacement + content.substring(inlineEdit.end)
    setContent(next)
    setDirty(next !== savedContentRef.current)
    setInlineEdit(null)
  }, [inlineEdit, content])

  const handleInlineEditOpen = useCallback((text: string, start: number, end: number, splitPx: number) => {
    setInlineEdit({ text, start, end, splitPx })
  }, [])

  const loaded = isCanvas ? canvasDoc !== null : content !== null
  if (!loaded) {
    return <div className="flex items-center justify-center py-6 text-xs text-ink-faint">Loading...</div>
  }

  const splitIdx = inlineEdit && content ? content.indexOf("\n", inlineEdit.end) : -1
  const splitChar = splitIdx === -1 ? (content?.length ?? 0) : splitIdx + 1
  const [topContent, botContent] = inlineEdit && content
    ? [content.substring(0, splitChar), content.substring(splitChar)]
    : ["", ""]
  const botStart = topContent.split("\n").length
  // Pixel height of the bottom slice so the scroll viewport knows its true extent
  const BOT_LINE_H = 19.5  // 0.75rem × 1.625
  const BOT_PAD = 16       // p-4
  const botHeight = BOT_PAD + botContent.split("\n").length * BOT_LINE_H + BOT_PAD

  const textBody = content !== null && !isCanvas && (
    // When inline-edit is active the div becomes a plain scroll viewport so the
    // user can scroll through the whole document with the panel inserted inline.
    <div className={`relative flex-1 min-h-0 group/editor${inlineEdit && model ? " overflow-y-auto" : " flex flex-col"}`}>
      <ViewPills view={view} onViewChange={setView} />
      {view === "edit" && inlineEdit && model ? (
        <>
          <div style={{ height: inlineEdit.splitPx }} className="flex flex-col shrink-0 overflow-hidden">
            <ConsoleEditor value={topContent} onChange={() => {}} language={language} readOnly />
          </div>
          <InlineEditPanel
            selectedText={inlineEdit.text}
            model={model}
            onApply={handleInlineApply}
            onClose={() => setInlineEdit(null)}
          />
          <div style={{ height: botHeight }} className="flex flex-col shrink-0">
            <ConsoleEditor value={botContent} onChange={() => {}} language={language} startLineNumber={botStart} readOnly />
          </div>
        </>
      ) : view === "edit" ? (
        <ConsoleEditor value={content} onChange={handleTextChange} language={language} placeholder="Start writing..." onInlineEdit={model ? handleInlineEditOpen : undefined} />
      ) : (
        // eslint-disable-next-line jsx-a11y/click-events-have-key-events, jsx-a11y/no-static-element-interactions
        <div className="flex-1 overflow-y-auto min-h-0 p-4 pt-10" onClick={handlePreviewClick}>
          <LaTeXMarkdown content={renderedContent ?? content} />
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
        className="absolute inset-0 z-30 flex flex-row bg-paper"
      >
        <div className="flex-1 min-w-0 flex flex-col relative">
          {editorBody}
          {model && !isCanvas && (
            <button
              onClick={() => setSidebarOpen(v => !v)}
              className="absolute top-2 right-3 z-10 p-1 rounded text-ink-subtle hover:text-ink hover:bg-surface/80 transition-colors"
            >
              {sidebarOpen ? <ChevronRight className="h-3 w-3" /> : <ChevronLeft className="h-3 w-3" />}
            </button>
          )}
        </div>
        {sidebarOpen && model && (
          <EditorSidebar
            content={content ?? ""}
            model={model}
            width={sidebarWidth}
            onWidthChange={setSidebarWidth}
            onApply={handleSidebarApply}
          />
        )}
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
