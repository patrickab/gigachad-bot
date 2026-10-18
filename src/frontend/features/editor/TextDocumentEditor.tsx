"use client"

import { useCallback, useEffect, useEffectEvent, useMemo, useRef, useState } from "react"
import { createPortal } from "react-dom"
import { Maximize2, Minimize2, X } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"
import { cn } from "@/lib/utils"
import { AssistedConsoleEditor } from "./AssistedConsoleEditor"
import { NeovimEditor, useNeovimTerminal } from "./NeovimEditor"
import { LaTeXMarkdown } from "@/components/content/LaTeXMarkdown"
import { PlotElement } from "@/components/content/PlotElement"
import type { PlotFigure } from "@/lib/domain/types"
import { ResizableEditor } from "./ResizableEditor"
import type { CanvasToolbarSlots } from "@/features/editor/canvas/CanvasToolbar"
import { loadFileViewerText, readFileRootRendered, writeArchitectureSource, writeDocument, writeRootFile, ApiError } from "@/lib/backend/api"
import { architectureSlug, architectureSource, isArchitecturePath } from "@/lib/domain/architecturePaths"
import { subscribeToChanges, type ChangeEvent } from "@/lib/backend/syncStream"
import { baseName } from "@/lib/paths"
import { useTabActive } from "@/contexts/TabContext"
import { useNeovimSetting } from "@/contexts/SettingsContext"

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
function isToolArtifact(path: string, content: string): boolean {
  return path.endsWith(".plot.json") || /```markmap\b/i.test(content)
}

function parsePlotFigure(content: string): PlotFigure | null {
  try {
    const figure: unknown = JSON.parse(content)
    return figure && typeof figure === "object" && !Array.isArray(figure) ? figure as PlotFigure : null
  } catch {
    return null
  }
}

function ViewPills({ view, onViewChange }: { view: EditorView; onViewChange: (v: EditorView) => void }) {
  return (
    <div role="group" aria-label="Show as" className="absolute top-2 left-3 z-10 flex items-center gap-px rounded-md bg-surface/80 backdrop-blur-sm p-0.5 opacity-0 hover:opacity-100 transition-opacity">
      {(["edit", "preview"] as const).map((v) => (
        <button
          key={v}
          type="button"
          aria-pressed={view === v}
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

/** C4 "Show as" switch for tool artifacts: rendered view first, source text second. */
function ShowAsToggle({ view, onViewChange }: { view: EditorView; onViewChange: (v: EditorView) => void }) {
  return (
    <div role="group" aria-label="Show as" className="flex gap-px rounded bg-surface p-px">
      {(["preview", "edit"] as const).map((option) => (
        <button key={option} type="button" aria-pressed={view === option} onClick={() => onViewChange(option)} className={cn("rounded px-1.5 py-px transition-colors", view === option ? "bg-surface-elevated text-ink" : "hover:text-ink")}>
          {option === "preview" ? "Render" : "Text"}
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
  const isActive = useTabActive()
  const neovim = useNeovimSetting()
  const [content, setContent] = useState<string | null>(null)
  const [renderedContent, setRenderedContent] = useState<string | null>(null)
  const [loadedPath, setLoadedPath] = useState<string | null>(null)
  const [loadError, setLoadError] = useState<"failed" | "binary" | null>(null)
  const [dirty, setDirty] = useState(false)
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [isFullscreen, setIsFullscreen] = useState(!!overlay)
  const [view, setView] = useState<EditorView>(() => path.endsWith(".plot.json") ? "preview" : "edit")
  const initializedViewPath = useRef<string | null>(null)
  const savedContentRef = useRef("")

  const filename = baseName(path)
  const language = editorLanguage(path)
  const isArtifact = isToolArtifact(path, content ?? "")
  const plotFigure = useMemo(() => path.endsWith(".plot.json") && content !== null ? parsePlotFigure(content) : null, [content, path])

  useEffect(() => {
    if (loadedPath !== path || content === null || initializedViewPath.current === path) return
    initializedViewPath.current = path
    setView(isToolArtifact(path, content) ? "preview" : "edit")
  }, [path, content, loadedPath])
  const modeLabelRef = useRef(onModeLabel)
  useEffect(() => { modeLabelRef.current = onModeLabel }, [onModeLabel])
  useEffect(() => {
    if (overlay) modeLabelRef.current?.(filename)
  }, [overlay, filename])

  const loadFromServer = useCallback(() => {
    const apply = (text: string) => {
      setLoadError(null)
      setContent(text)
      setLoadedPath(path)
      savedContentRef.current = text
    }
    // A confirmed-missing document legitimately starts blank. Any other failure must
    // not arm a save of blank content over the real stored content.
    return loadFileViewerText(path).then(apply).catch((err) => {
      if (err instanceof ApiError && err.status === 404) apply("")
      else setLoadError(err instanceof ApiError && err.status === 415 ? "binary" : "failed")
    })
  }, [path])

  const [renderedScope, setRenderedScope] = useState({ path, overlay })
  // Drop the rendered preview of the previous document or mode
  if (renderedScope.path !== path || renderedScope.overlay !== overlay) {
    setRenderedScope({ path, overlay })
    setRenderedContent(null)
  }

  useEffect(() => {
    loadFromServer()
    if (overlay) readFileRootRendered(path).then(setRenderedContent).catch(() => {})
  }, [path, overlay, loadFromServer])

  const handleTextChange = useCallback((value: string) => {
    setContent(value)
    setDirty(value !== savedContentRef.current)
  }, [])

  // Reload on remote changes unless there are unsaved local edits. `resource_key` may be a path suffix.
  const reloadOnRemoteChange = useEffectEvent((event: ChangeEvent) => {
    if (dirty || event.resource_kind !== "document") return
    if (event.resource_key !== path && !path.endsWith(`/${event.resource_key}`)) return
    loadFromServer()
  })
  useEffect(() => subscribeToChanges((event) => reloadOnRemoteChange(event)), [])

  const clearLiveContent = useEffectEvent((livePath: string) => { onLiveContent?.(livePath, null) })
  useEffect(() => () => { clearLiveContent(path) }, [path])

  const persistNow = useCallback(async (next: string) => {
    if (persistOverride) await persistOverride(next)
    // An absolute path is a file mount file, referenced by a project or opened directly: save it in place.
    else if (path.startsWith("/")) await writeRootFile(path, next)
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

  // Neovim owns the text while it runs: `:w` persists, and React shows what was stored.
  const saveFromNeovim = useCallback(async (text: string) => {
    await persist(text)
    setContent(savedContentRef.current)
    return savedContentRef.current
  }, [persist])
  const neovimSession = useNeovimTerminal(
    neovim.enabled && content !== null && loadedPath === path ? { key: path, path, filename, content } : null,
    saveFromNeovim,
    onClose,
  )

  const handleSave = useCallback(async () => {
    if (content === null || saving) return
    setSaving(true)
    setSaveError(null)
    try { await persist(content) } catch (err) { setSaveError(err instanceof Error ? err.message : "Save failed") }
    setSaving(false)
  }, [content, saving, persist])

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (!isActive) return
      if (!neovim.enabled && (event.ctrlKey || event.metaKey) && (event.key === "s" || event.key === "Enter")) {
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
  }, [handleSave, isFullscreen, overlay, onClose, isActive, neovim.enabled])

  const handlePreviewClick = useCallback((e: React.MouseEvent) => {
    const anchor = (e.target as HTMLElement).closest("a")
    if (!anchor) return
    const href = anchor.getAttribute("href")
    if (!href?.startsWith("#file:")) return
    e.preventDefault()
    onNavigate?.(decodeURIComponent(href.slice("#file:".length)))
  }, [onNavigate])

  if (loadError === "binary") {
    return <div className="flex items-center justify-center py-6 text-xs text-ink-faint">This is not a text file, so it cannot be opened in the editor.</div>
  }

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

  // After a path change, `content` still belongs to the previous document until the new load lands.
  if (content === null || loadedPath !== path) {
    return <div className="flex items-center justify-center py-6 text-xs text-ink-faint">Loading...</div>
  }

  const preview = (
    <div className="flex-1 overflow-y-auto min-h-0 p-4 pt-10" onClick={handlePreviewClick}>
      {path.endsWith(".plot.json")
        ? plotFigure ? <PlotElement figure={plotFigure} /> : <p className="text-xs text-danger">This plot file is not valid Plotly JSON.</p>
        : <LaTeXMarkdown content={renderedContent ?? content} />}
    </div>
  )

  // Neovim stays mounted while previewing, so switching views keeps unsaved edits.
  const editorBody = neovim.enabled ? (
    <div className="relative flex-1 min-h-0 group/editor flex flex-col">
      {!isArtifact && <ViewPills view={view} onViewChange={setView} />}
      <NeovimEditor terminal={neovimSession.terminal} status={neovimSession.status} onRestart={neovimSession.restart} className={view === "edit" ? undefined : "hidden"} />
      {view === "preview" && preview}
    </div>
  ) : view === "edit" ? (
    <AssistedConsoleEditor value={content} onChange={handleTextChange} language={language} model={model} placeholder="Start writing..." className="group/editor">
      {!isArtifact && <ViewPills view={view} onViewChange={setView} />}
    </AssistedConsoleEditor>
  ) : (
    <div className="relative flex-1 min-h-0 group/editor flex flex-col">
      {!isArtifact && <ViewPills view={view} onViewChange={setView} />}
      {preview}
    </div>
  )

  // Sidebar documents: C4-style header with filename, view switch and window controls.
  const chrome = !overlay && (
    <div className="flex h-6 shrink-0 items-center gap-2 border-b border-divider/50 px-2 text-[10px] text-ink-faint">
      <span className="min-w-0 flex-1 truncate">{filename}{dirty && " (modified)"}</span>
      {isArtifact && <ShowAsToggle view={view} onViewChange={setView} />}
      <button aria-label={isFullscreen ? "Exit document fullscreen" : "Enter document fullscreen"} onClick={() => setIsFullscreen((f) => !f)} className="rounded p-0.5 hover:text-ink">
        {isFullscreen ? <Minimize2 className="h-3 w-3" /> : <Maximize2 className="h-3 w-3" />}
      </button>
      <button aria-label="Close document" onClick={onClose} className="rounded p-0.5 hover:text-danger">
        <X className="h-3 w-3" />
      </button>
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
            {isArtifact && <ShowAsToggle view={view} onViewChange={setView} />}
          </div>
        </div>
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
