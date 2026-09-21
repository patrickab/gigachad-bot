"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { Download, Save, Upload, X } from "lucide-react"
import { ArchitectureGraphSurface } from "./ArchitectureGraphSurface"
import { MermaidImportModal } from "./MermaidImportModal"
import {
  parseArchitectureGraph,
  serializeArchitectureGraph,
  type ArchitectureGraph,
} from "@/lib/architectureGraph"
import { graphToMermaidFlowchart, parseMermaidFlowchart } from "@/lib/mermaidGraph"
import { readArchitectureGraph, writeArchitectureGraph } from "@/lib/api"
import { useGraphAutosave } from "@/lib/graphAutosave"
import { subscribeToChanges } from "@/lib/syncStream"
import { cn } from "@/lib/utils"

interface ArchitectureGraphEditorProps {
  path: string
  overlay?: boolean
  onClose: () => void
  onSaved?: (filename?: string, content?: string) => void
  onModeLabel?: (label: string) => void
}

type View = "diagram" | "source"

// Edits autosave: the surface commits drafts as you type, and this view debounces
// them to disk with a bounded max wait so sustained typing still reaches the file.
export function ArchitectureGraphEditor({ path, overlay = false, onClose, onSaved, onModeLabel }: ArchitectureGraphEditorProps) {
  const name = path.split("/").pop() ?? path
  const [graph, setGraph] = useState<ArchitectureGraph | null>(null)
  const [source, setSource] = useState("")
  const [view, setView] = useState<View>("diagram")
  const [error, setError] = useState<string | null>(null)
  const onSavedRef = useRef(onSaved)
  onSavedRef.current = onSaved

  const write = useCallback((content: string) => writeArchitectureGraph(name, content), [name])
  const handleSaved = useCallback((content: string) => onSavedRef.current?.(name, content), [name])
  const handleError = useCallback(() => setError("Could not save Architecture Graph"), [])
  const { queue, flush, cancel, markSaved, dirty } = useGraphAutosave({ key: name, write, onSaved: handleSaved, onError: handleError })

  const load = useCallback(async () => {
    const document = await readArchitectureGraph(name)
    setGraph(parseArchitectureGraph(document.content))
    setSource(document.content)
    markSaved(document.content)
  }, [name, markSaved])

  useEffect(() => {
    let active = true
    setGraph(null)
    setError(null)
    load().catch((cause: unknown) => {
      if (active) setError(cause instanceof Error ? cause.message : "Could not load Architecture Graph")
    })
    return () => { active = false }
  }, [load])

  // A remote edit only replaces local state when nothing is unsaved here, so a
  // notification can never discard an edit that has not reached the server yet.
  const dirtyRef = useRef(dirty)
  dirtyRef.current = dirty

  useEffect(() => subscribeToChanges((event) => {
    if (dirtyRef.current) return
    if (event.resource_kind !== "document" || !event.resource_key.endsWith(`/${name}`)) return
    load().catch(() => setError("Could not reload Architecture Graph"))
  }), [name, load])

  useEffect(() => {
    if (!overlay) return
    onModeLabel?.(name)
  }, [name, onModeLabel, overlay])

  const applyGraph = useCallback((next: ArchitectureGraph) => {
    const nextSource = serializeArchitectureGraph(next)
    setGraph(next)
    setSource(nextSource)
    setError(null)
    queue(nextSource)
  }, [queue])
  const [mermaidOpen, setMermaidOpen] = useState(false)
  const [mermaidCopied, setMermaidCopied] = useState(false)
  const handleMermaidImport = useCallback((mermaidSource: string): string | null => {
    try {
      applyGraph(parseMermaidFlowchart(mermaidSource, graph?.title ?? name))
      setMermaidOpen(false)
      return null
    } catch (cause) {
      return cause instanceof Error ? cause.message : "Invalid Mermaid flowchart"
    }
  }, [applyGraph, graph?.title, name])
  const handleMermaidExport = useCallback(() => {
    if (!graph) return
    navigator.clipboard?.writeText(graphToMermaidFlowchart(graph)).then(() => {
      setMermaidCopied(true)
      setTimeout(() => setMermaidCopied(false), 1500)
    })
  }, [graph])

  const handleSourceChange = useCallback((nextSource: string) => {
    setSource(nextSource)
    try {
      const next = parseArchitectureGraph(nextSource)
      setGraph(next)
      setError(null)
      queue(nextSource)
    } catch (cause) {
      // Keep the broken buffer editable; the file keeps its last valid content.
      cancel()
      setError(cause instanceof Error ? cause.message : "Invalid Architecture Graph YAML")
    }
  }, [queue, cancel])

  const handleSave = useCallback(() => {
    try {
      const next = parseArchitectureGraph(source)
      const normalized = serializeArchitectureGraph(next)
      setGraph(next)
      setSource(normalized)
      setError(null)
      queue(normalized)
      flush()
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Invalid Architecture Graph YAML")
    }
  }, [queue, flush, source])

  const showView = useCallback((next: View) => {
    // Land pending work before the buffer behind it is swapped out.
    flush()
    setView(next)
  }, [flush])

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
        event.preventDefault()
        handleSave()
      }
    }
    document.addEventListener("keydown", onKeyDown)
    return () => document.removeEventListener("keydown", onKeyDown)
  }, [handleSave])

  if (!graph) {
    return <div className="flex min-h-[280px] items-center justify-center text-xs text-ink-faint">{error ?? "Loading Architecture Graph..."}</div>
  }

  return (
    <div className={cn("flex min-h-[360px] min-w-0 flex-col bg-paper", overlay && "absolute inset-0 z-30")}>
      <header className="flex h-10 shrink-0 items-center gap-3 border-b border-divider px-3">
        <span className="min-w-0 flex-1 truncate text-xs font-semibold text-ink">{name}</span>
        <div className="flex items-center gap-1" role="tablist" aria-label="Architecture Graph view">
          {(["diagram", "source"] as const).map((candidate) => (
            <button
              key={candidate}
              type="button"
              role="tab"
              aria-selected={view === candidate}
              onClick={() => showView(candidate)}
              className={cn("rounded px-2 py-1 text-[11px] transition-colors", view === candidate ? "bg-surface-elevated text-ink" : "text-ink-faint hover:text-ink")}
            >
              {candidate === "diagram" ? "Diagram" : "Source"}
            </button>
          ))}
        </div>
        {dirty && <span className="text-[10px] text-ink-faint">saving…</span>}
        <button type="button" onClick={() => setMermaidOpen(true)} className="rounded p-1 text-ink-subtle hover:bg-hover hover:text-ink" aria-label="Import from Mermaid"><Upload className="h-3.5 w-3.5" /></button>
        <button type="button" onClick={handleMermaidExport} className="rounded p-1 text-ink-subtle hover:bg-hover hover:text-ink" aria-label="Copy as Mermaid">
          {mermaidCopied ? <span className="text-[10px] text-ink-muted">Copied</span> : <Download className="h-3.5 w-3.5" />}
        </button>
        <button type="button" onClick={handleSave} className="rounded p-1 text-ink-subtle hover:bg-hover hover:text-ink" aria-label="Save Architecture Graph"><Save className="h-3.5 w-3.5" /></button>
        <button type="button" onClick={onClose} className="rounded p-1 text-ink-subtle hover:bg-hover hover:text-danger" aria-label="Close"><X className="h-3.5 w-3.5" /></button>
      </header>
      <MermaidImportModal open={mermaidOpen} onClose={() => setMermaidOpen(false)} onImport={handleMermaidImport} />
      {error && <p className="shrink-0 border-b border-divider bg-surface px-3 py-1.5 text-xs text-danger" role="alert">{error}</p>}
      <div className="min-h-0 flex-1">
        {view === "diagram" ? (
          <ArchitectureGraphSurface graph={graph} onChange={applyGraph} className="h-full" />
        ) : (
          <textarea
            aria-label="Architecture Graph YAML source"
            value={source}
            onChange={(event) => handleSourceChange(event.target.value)}
            spellCheck={false}
            className="h-full w-full resize-none bg-paper p-4 font-mono text-xs leading-6 text-ink outline-none"
          />
        )}
      </div>
    </div>
  )
}
