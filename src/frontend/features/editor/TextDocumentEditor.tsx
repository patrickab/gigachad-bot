"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { createPortal } from "react-dom"
import { ChevronLeft, ChevronRight, Maximize2, Minimize2, Save, X } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"
import { cn } from "@/lib/utils"
import { AssistedConsoleEditor } from "./AssistedConsoleEditor"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"
import { ResizableEditor } from "./ResizableEditor"
import type { CanvasToolbarSlots } from "@/features/editor/canvas/CanvasToolbar"
import { loadFileViewerText, readFileRootRendered, writeArchitectureSource, writeDocument, ApiError } from "@/lib/backend/api"
import { EditorSidebar } from "./EditorSidebar"
import { architectureSlug, architectureSource, isArchitecturePath } from "@/lib/domain/architecturePaths"
import { subscribeToChanges } from "@/lib/backend/syncStream"
import { baseName } from "@/lib/paths"

type EditorView = "edit" | "preview"

export interface DocumentEditorProps {
  path: string
  slug: string
  onClose: () => void
  onSaved?: (filename?: string, content?: string) => void
  onLiveContent?: (path: string, content: string | null) => void
  overlay?: boolean
  persistOverride?: (content: string) => Promise<void>
  onModeLabel?: (label: string) => void
  onNavigate?: (path: string) => void
  model?: string
  canvasToolbarSlots?: CanvasToolbarSlots | null
}

function editorLanguage(path: string): string {
  if (path.endsWith(".tex")) return "latex"
  if (path.endsWith(".c4")) return "likec4"
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

/** Edit a workspace's `.c4` source as text; the architecture route refuses text that no longer parses into a model. */
export function ArchitectureSourceEditor(props: DocumentEditorProps) {
  const slug = architectureSlug(props.path)
  const file = architectureSource(props.path)
  return <TextDocumentEditor {...props} persistOverride={async (content) => { await writeArchitectureSource(slug, file, content) }} />
}

/** Text document editor; `persistOverride` replaces the default document write. */
export function TextDocumentEditor({ path, slug, onClose, onSaved, onLiveContent, overlay, persistOverride, onModeLabel, onNavigate, model }: DocumentEditorProps) {
  const [content, setContent] = useState<string | null>(null)
  const [renderedContent, setRenderedContent] = useState<string | null>(null)
  const [loadError, setLoadError] = useState(false)
  const [dirty, setDirty] = useState(false)
  const dirtyRef = useRef(dirty)
  dirtyRef.current = dirty
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [isFullscreen, setIsFullscreen] = useState(!!overlay)
  const [view, setView] = useState<EditorView>("edit")
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [sidebarWidth, setSidebarWidth] = useState(340)
  const savedContentRef = useRef("")

  const filename = baseName(path)
  const language = editorLanguage(path)

  const modeLabelRef = useRef(onModeLabel)
  useEffect(() => { modeLabelRef.current = onModeLabel }, [onModeLabel])
  useEffect(() => {
    if (overlay) modeLabelRef.current?.(filename)
  }, [overlay, filename])

  const loadFromServer = useCallback(() => {
    const apply = (text: string) => {
      setLoadError(false)
      setContent(text)
      savedContentRef.current = text
    }
    // A confirmed-missing document legitimately starts blank. Any other failure must
    // not arm a save of blank content over the real stored content.
    return loadFileViewerText(path).then(apply).catch((err) => {
      if (err instanceof ApiError && err.status === 404) apply("")
      else setLoadError(true)
    })
  }, [path])

  useEffect(() => {
    setRenderedContent(null)
    loadFromServer()
    if (overlay) readFileRootRendered(path).then(setRenderedContent).catch(() => {})
  }, [path, overlay, loadFromServer])

  const handleTextChange = useCallback((value: string) => {
    setContent(value)
    setDirty(value !== savedContentRef.current)
  }, [])

  // Reload on remote changes unless there are unsaved local edits. `resource_key` may be a path suffix.
  useEffect(() => subscribeToChanges((event) => {
    if (dirtyRef.current || event.resource_kind !== "document") return
    if (event.resource_key !== path && !path.endsWith(`/${event.resource_key}`)) return
    loadFromServer()
  }), [path, loadFromServer])

  const liveRef = useRef(onLiveContent)
  liveRef.current = onLiveContent
  useEffect(() => () => { liveRef.current?.(path, null) }, [path])

  const persistNow = useCallback(async (next: string) => {
    if (persistOverride) await persistOverride(next)
    else await writeDocument(slug, filename, next)
    savedContentRef.current = next
    setDirty(false)
    // The server prunes uses of removed elements from a hand-edited `.c4`; show what it stored.
    if (isArchitecturePath(path)) {
      const stored = await loadFileViewerText(path).catch(() => next)
      if (stored !== next) { setContent(stored); savedContentRef.current = stored }
    }
    onSaved?.(filename, next)
  }, [slug, filename, path, onSaved, persistOverride])

  // Saves are chained so an earlier slow write can never resolve after,
  // and silently clobber, a newer one.
  const saveChain = useRef(Promise.resolve())
  const persist = useCallback((next: string) => {
    const run = saveChain.current.then(() => persistNow(next))
    saveChain.current = run.catch(() => {})
    return run
  }, [persistNow])

  const handleSave = useCallback(async () => {
    if (content === null || saving) return
    setSaving(true)
    setSaveError(null)
    try { await persist(content) } catch (err) { setSaveError(err instanceof Error ? err.message : "Save failed") }
    setSaving(false)
  }, [content, saving, persist])

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && (event.key === "s" || event.key === "Enter")) {
        event.preventDefault()
        handleSave()
      }
      if (event.key === "Escape") {
        if (overlay) { event.preventDefault(); onClose() }
        else if (isFullscreen) { event.preventDefault(); setIsFullscreen(false) }
      }
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [handleSave, isFullscreen, overlay, onClose])

  const handlePreviewClick = useCallback((e: React.MouseEvent) => {
    const anchor = (e.target as HTMLElement).closest("a")
    if (!anchor) return
    const href = anchor.getAttribute("href")
    if (!href?.startsWith("#file:")) return
    e.preventDefault()
    onNavigate?.(decodeURIComponent(href.slice("#file:".length)))
  }, [onNavigate])

  const handleSidebarApply = useCallback((newContent: string) => {
    setContent(newContent)
    setDirty(newContent !== savedContentRef.current)
  }, [])

  if (loadError) {
    return (
      <div className="flex flex-col items-center justify-center gap-2 py-6 text-xs text-ink-faint">
        <span>Failed to load this document. Its stored content was left untouched.</span>
        <button onClick={() => loadFromServer()} className="rounded px-2 py-1 text-ink-subtle hover:text-ink hover:bg-hover transition-colors">
          Retry
        </button>
      </div>
    )
  }

  if (content === null) {
    return <div className="flex items-center justify-center py-6 text-xs text-ink-faint">Loading...</div>
  }

  const editorBody = view === "edit" ? (
    <AssistedConsoleEditor value={content} onChange={handleTextChange} language={language} model={model} placeholder="Start writing..." className="group/editor">
      <ViewPills view={view} onViewChange={setView} />
    </AssistedConsoleEditor>
  ) : (
    <div className="relative flex-1 min-h-0 group/editor flex flex-col">
      <ViewPills view={view} onViewChange={setView} />
      {/* eslint-disable-next-line jsx-a11y/click-events-have-key-events, jsx-a11y/no-static-element-interactions */}
      <div className="flex-1 overflow-y-auto min-h-0 p-4 pt-10" onClick={handlePreviewClick}>
        <LaTeXMarkdown content={renderedContent ?? content} />
      </div>
    </div>
  )

  // Inline chrome (sidebar documents): filename, save, fullscreen, close.
  const chrome = !overlay && (
    <div className="flex items-center justify-between px-3 py-1.5 border-b border-divider/50 shrink-0">
      <div className="flex items-center gap-2 min-w-0">
        <span className="text-[11px] font-medium text-ink truncate">{filename}</span>
        {dirty && <span className="text-[10px] text-ink-faint">(modified)</span>}
      </div>
      <div className="flex items-center gap-1 shrink-0">
        <button aria-label="Save document" onClick={handleSave} disabled={!dirty || saving} className="rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover disabled:opacity-30 transition-colors">
          <Save className="h-3.5 w-3.5" />
        </button>
        <button aria-label={isFullscreen ? "Exit document fullscreen" : "Enter document fullscreen"} onClick={() => setIsFullscreen((f) => !f)} className="rounded p-1 text-ink-subtle hover:text-ink hover:bg-hover transition-colors">
          {isFullscreen ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
        </button>
        <button aria-label="Close document" onClick={onClose} className="rounded p-1 text-ink-subtle hover:text-danger transition-colors">
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
          <div className="absolute top-2 right-3 z-10 flex items-center gap-1">
            {saveError && <span role="alert" className="max-w-[min(60ch,60vw)] text-right text-[10px] leading-snug text-danger">{saveError}</span>}
            <button aria-label="Save document" onClick={handleSave} disabled={!dirty || saving} className="rounded p-1 text-ink-subtle hover:text-ink hover:bg-surface/80 disabled:opacity-30 transition-colors">
              <Save className="h-3.5 w-3.5" />
            </button>
            {model && (
              <button
                aria-label={sidebarOpen ? "Hide assistant" : "Show assistant"}
                onClick={() => setSidebarOpen(v => !v)}
                className="p-1 rounded text-ink-subtle hover:text-ink hover:bg-surface/80 transition-colors"
              >
                {sidebarOpen ? <ChevronRight className="h-3 w-3" /> : <ChevronLeft className="h-3 w-3" />}
              </button>
            )}
          </div>
        </div>
        {sidebarOpen && model && (
          <EditorSidebar
            content={content}
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
