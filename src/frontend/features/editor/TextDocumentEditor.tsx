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
import { readFileRootRendered, writeArchitectureSource, writeDocument, writeRootFile } from "@/lib/backend/api"
import { architectureSlug, architectureSource, isArchitecturePath } from "@/lib/domain/architecturePaths"
import { baseName } from "@/lib/paths"
import { useTabActive } from "@/contexts/TabContext"
import { useNeovimSetting } from "@/contexts/SettingsContext"
import { useRootFileDocument } from "./useRootFileDocument"
import { useAsyncAction } from "@/hooks/useAsyncAction"
import { useEventListener } from "@/hooks/useEventListener"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import { useResetOnChange } from "@/hooks/useResetOnChange"
import { LoadFailure, LoadingPlaceholder } from "@/components/ui/DocumentLoadState"

type EditorView = "edit" | "preview"

export interface DocumentEditorProps {
  path: string
  slug: string
  onClose: () => void
  onSaved?: () => void
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

function ViewSwitch({ view, onViewChange, options, className, buttonClass, inactiveClass }: {
  view: EditorView
  onViewChange: (v: EditorView) => void
  options: readonly (readonly [EditorView, string])[]
  className: string
  buttonClass: string
  inactiveClass: string
}) {
  return (
    <div role="group" aria-label="Show as" className={className}>
      {options.map(([option, label]) => (
        <button
          key={option}
          type="button"
          aria-pressed={view === option}
          onClick={() => onViewChange(option)}
          className={cn(buttonClass, view === option ? "bg-surface-elevated text-ink" : inactiveClass)}
        >
          {label}
        </button>
      ))}
    </div>
  )
}

function ViewPills({ view, onViewChange }: { view: EditorView; onViewChange: (v: EditorView) => void }) {
  return (
    <ViewSwitch
      view={view}
      onViewChange={onViewChange}
      options={[["edit", "Edit"], ["preview", "Preview"]]}
      className="absolute top-2 left-3 z-10 flex items-center gap-px rounded-md bg-surface/80 backdrop-blur-sm p-0.5 opacity-0 hover:opacity-100 transition-opacity"
      buttonClass="px-2 py-0.5 rounded text-[10px] font-medium transition-colors"
      inactiveClass="text-ink-subtle hover:text-ink"
    />
  )
}

/** C4 "Show as" switch for tool artifacts: rendered view first, source text second. */
function ShowAsToggle({ view, onViewChange }: { view: EditorView; onViewChange: (v: EditorView) => void }) {
  return (
    <ViewSwitch
      view={view}
      onViewChange={onViewChange}
      options={[["preview", "Render"], ["edit", "Text"]]}
      className="flex gap-px rounded bg-surface p-px"
      buttonClass="rounded px-1.5 py-px transition-colors"
      inactiveClass="hover:text-ink"
    />
  )
}

/** Edit a workspace's `.c4` source as text; the architecture route refuses text that no longer parses into a model. */
export function ArchitectureSourceEditor(props: DocumentEditorProps) {
  const slug = architectureSlug(props.path)
  const file = architectureSource(props.path)
  return <TextDocumentEditor {...props} persistOverride={async (content) => { await writeArchitectureSource(slug, file, content) }} />
}

/** Text document editor; `persistOverride` replaces the default document write. */
export function TextDocumentEditor({ path, slug, onClose, onSaved, overlay, persistOverride, onModeLabel, onNavigate, model }: DocumentEditorProps) {
  const isActive = useTabActive()
  const neovim = useNeovimSetting()
  const [renderedContent, setRenderedContent] = useState<string | null>(null)
  const [isFullscreen, setIsFullscreen] = useState(!!overlay)
  const [view, setView] = useState<EditorView>(() => path.endsWith(".plot.json") ? "preview" : "edit")
  const initializedViewPath = useRef<string | null>(null)

  const filename = baseName(path)
  const write = useCallback(async (next: string) => {
    if (persistOverride) await persistOverride(next)
    // An absolute path is a file mount file, referenced by a project or opened directly: save it in place.
    else if (path.startsWith("/")) await writeRootFile(path, next)
    else await writeDocument(slug, filename, next)
  }, [slug, filename, path, persistOverride])
  // A confirmed-missing document legitimately starts blank. The server prunes uses of removed elements
  // from a hand-edited `.c4`, so its saves read back what it stored.
  const { content, loadedPath, loadError, dirty, reload: loadFromServer, handleTextChange, persist, saveFromNeovim } = useRootFileDocument({
    path, write, blankWhenMissing: true, readBackAfterSave: isArchitecturePath(path), onSaved,
  })

  const language = editorLanguage(path)
  const isArtifact = isToolArtifact(path, content ?? "")
  const plotFigure = useMemo(() => path.endsWith(".plot.json") && content !== null ? parsePlotFigure(content) : null, [content, path])

  useEffect(() => {
    if (loadedPath !== path || content === null || initializedViewPath.current === path) return
    initializedViewPath.current = path
    setView(isToolArtifact(path, content) ? "preview" : "edit")
  }, [path, content, loadedPath])
  const announceMode = useEffectEvent(() => { if (overlay) onModeLabel?.(filename) })
  useEffect(() => { announceMode() }, [overlay, filename])

  // Drop the rendered preview of the previous document or mode
  useResetOnChange(overlay ? path : null, () => setRenderedContent(null))

  useEffect(() => {
    loadFromServer()
    if (overlay) readFileRootRendered(path).then(setRenderedContent).catch(() => {})
  }, [path, overlay, loadFromServer])

  const neovimSession = useNeovimTerminal(
    neovim.enabled && content !== null && loadedPath === path ? { key: path, path, filename, content } : null,
    saveFromNeovim,
    onClose,
  )

  const { run: save, pending: saving, error: saveError } = useAsyncAction(persist, "Save failed")
  const handleSave = () => {
    if (content !== null && !saving) void save(content)
  }

  useEventListener("document", "keydown", (event) => {
    if (!isActive) return
    if (!neovim.enabled && (event.ctrlKey || event.metaKey) && (event.key === "s" || event.key === "Enter")) {
      event.preventDefault()
      handleSave()
    }
  })

  useEscapeLayer(() => { if (overlay) onClose(); else setIsFullscreen(false) }, isActive && (!!overlay || isFullscreen))

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

  if (loadError) return <LoadFailure message="Failed to load this document. Its stored content was left untouched." onRetry={loadFromServer} />

  // After a path change, `content` still belongs to the previous document until the new load lands.
  if (content === null || loadedPath !== path) return <LoadingPlaceholder />

  const pills = !isArtifact && <ViewPills view={view} onViewChange={setView} />

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
      {pills}
      <NeovimEditor terminal={neovimSession.terminal} status={neovimSession.status} onRestart={neovimSession.restart} className={view === "edit" ? undefined : "hidden"} />
      {view === "preview" && preview}
    </div>
  ) : view === "edit" ? (
    <AssistedConsoleEditor value={content} onChange={handleTextChange} language={language} model={model} placeholder="Start writing..." className="group/editor">
      {pills}
    </AssistedConsoleEditor>
  ) : (
    <div className="relative flex-1 min-h-0 group/editor flex flex-col">
      {pills}
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
