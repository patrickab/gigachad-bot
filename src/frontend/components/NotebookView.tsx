"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { BookOpen, Play, RefreshCw, Save } from "lucide-react"
import { cn } from "@/lib/utils"
import { ApiError, fetchNotebook, runNotebook, saveNotebook, type NotebookCellOutput, type NotebookOutputs } from "@/lib/api"
import { parseNotebook, serializeNotebook, type NotebookCell } from "@/lib/notebook"
import { ConsoleEditor } from "./ConsoleEditor"
import { LaTeXMarkdown } from "./LaTeXMarkdown"
import { InlineEditPanel } from "./EditorSidebar"
import { PlotElement } from "./PlotElement"
import { useSettings } from "@/contexts/SettingsContext"

/** sha256 hex digest of a cell's source — the key the outputs sidecar is keyed by.
 * Read lazily: jsdom and non-secure contexts ship no WebCrypto, and the module
 * may be imported before a test stubs the global. */
export async function cellOutputKey(source: string): Promise<string> {
  const subtle = typeof crypto !== "undefined" ? crypto.subtle : undefined
  if (!subtle) return ""
  const digest = await subtle.digest("SHA-256", new TextEncoder().encode(source))
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("")
}

const SAVE_DEBOUNCE_MS = 1000
// Line metrics matching ConsoleEditor's 0.75rem × 1.625 text plus p-4 padding.
const CELL_LINE_H = 19.5
const CELL_PAD = 16

/** A pending Ctrl+I panel: `cellIndex === -1` means the whole-file Edit mode. */
interface InlineEditState {
  cellIndex: number
  text: string
  start: number
  end: number
  splitPx: number
}

interface NotebookViewProps {
  chatId: string
  /** Re-fetch when a tool edit or undo produces a new notebook revision. */
  refreshKey?: string
  /** True when the chat is known to have a notebook (mode toggle) — shows the
   * view while the first fetch is still in flight. */
  exists?: boolean
  /** Fires as fetches confirm whether a notebook exists. */
  onOpenNotebookChanged?: (exists: boolean) => void
}

function NotebookModePills({ mode, onModeChange }: { mode: "edit" | "notebook"; onModeChange: (m: "edit" | "notebook") => void }) {
  return (
    <div className="flex items-center gap-px rounded-md bg-surface/80 p-0.5">
      {(["edit", "notebook"] as const).map((m) => (
        <button
          key={m}
          type="button"
          onClick={() => onModeChange(m)}
          className={cn(
            "px-2 py-0.5 rounded text-[10px] font-medium transition-colors",
            mode === m ? "bg-surface-elevated text-ink" : "text-ink-subtle hover:text-ink",
          )}
        >
          {m === "edit" ? "Edit" : "Notebook"}
        </button>
      ))}
    </div>
  )
}

/** A cell's outputs below its editor; stale ones dim to half strength. */
function CellOutputList({ outputs, stale }: { outputs: NotebookCellOutput[]; stale: boolean }) {
  if (outputs.length === 0) return null
  return (
    <div className={cn("space-y-1 border-t border-divider/40 bg-paper/60 px-3 py-2 transition-opacity", stale && "opacity-50")}>
      {outputs.map((out, i) => {
        if (out.type === "plotly" && out.figure) return <PlotElement key={i} figure={out.figure} />
        return (
          <pre
            key={i}
            className={cn("whitespace-pre-wrap font-mono text-[10px] leading-relaxed", out.type === "error" ? "text-danger" : "text-ink-muted")}
          >
            {out.text}
          </pre>
        )
      })}
      {stale && <div className="text-[10px] text-ink-faint">stale — edited since this ran</div>}
    </div>
  )
}

/** Height a cell editor needs to show its whole source without inner scroll. */
function editorHeight(source: string): number {
  return CELL_PAD + Math.max(1, source.split("\n").length) * CELL_LINE_H + CELL_PAD
}

/** One cell card: markdown renders LaTeXMarkdown until clicked; code always
 * edits in place. Ctrl+I reuses the document editor's split-pane optic. */
function NotebookCellCard({
  cell,
  index,
  editing,
  outputs,
  stale,
  onSourceChange,
  onToggleEditing,
  onRun,
  running,
  onInlineEdit,
  inlineEdit,
  onInlineApply,
  onInlineClose,
  model,
}: {
  cell: NotebookCell
  index: number
  editing: boolean
  outputs: NotebookCellOutput[] | null
  stale: boolean
  onSourceChange: (source: string) => void
  onToggleEditing: () => void
  onRun: () => void
  running: boolean
  onInlineEdit: (text: string, start: number, end: number, splitPx: number) => void
  inlineEdit: { text: string; start: number; end: number; splitPx: number } | null
  onInlineApply: (replacement: string) => void
  onInlineClose: () => void
  model: string
}) {
  // Split the source around the selection end so the panel sits inline.
  const splitIdx = inlineEdit ? cell.source.indexOf("\n", inlineEdit.end) : -1
  const splitChar = splitIdx === -1 ? cell.source.length : splitIdx + 1
  const [topSource, botSource] = inlineEdit
    ? [cell.source.substring(0, splitChar), cell.source.substring(splitChar)]
    : ["", ""]

  return (
    <div className="rounded-lg border border-divider/40 bg-paper">
      <div className="flex items-center justify-between px-2 py-1">
        {cell.kind === "markdown" ? (
          <button
            type="button"
            onClick={onToggleEditing}
            aria-label={`markdown cell ${index + 1}`}
            className="text-[10px] font-medium text-ink-faint hover:text-ink-subtle transition-colors"
          >
            md [{index + 1}]
          </button>
        ) : (
          <span className="text-[10px] font-medium text-ink-faint">py [{index + 1}]</span>
        )}
        {cell.kind === "code" && (
          <button
            type="button"
            onClick={onRun}
            disabled={running}
            aria-label={`Run cell ${index + 1}`}
            className="rounded p-1 text-ink-subtle hover:text-ink hover:bg-surface-elevated disabled:opacity-30 transition-colors"
          >
            {running ? (
              <span className="block h-3 w-3 animate-spin rounded-full border-2 border-ink-faint border-t-ink" />
            ) : (
              <Play className="h-3 w-3" />
            )}
          </button>
        )}
      </div>
      {cell.kind === "markdown" && !editing ? (
        // Click the rendered body to swap it for the source editor.
        // eslint-disable-next-line jsx-a11y/click-events-have-key-events, jsx-a11y/no-static-element-interactions
        <div className="cursor-pointer px-3 pb-3" onClick={onToggleEditing}>
          <LaTeXMarkdown content={cell.source} compact />
        </div>
      ) : inlineEdit && model ? (
        <>
          <div style={{ height: inlineEdit.splitPx }} className="flex flex-col shrink-0 overflow-hidden">
            <ConsoleEditor value={topSource} onChange={() => {}} language="python" readOnly />
          </div>
          <InlineEditPanel selectedText={inlineEdit.text} model={model} onApply={onInlineApply} onClose={onInlineClose} />
          <div style={{ height: editorHeight(botSource) }} className="flex flex-col shrink-0">
            <ConsoleEditor
              value={botSource}
              onChange={() => {}}
              language="python"
              readOnly
              startLineNumber={topSource.split("\n").length}
            />
          </div>
        </>
      ) : (
        <div style={{ height: editorHeight(cell.source) }} className="flex flex-col">
          <ConsoleEditor
            value={cell.source}
            onChange={onSourceChange}
            language="python"
            placeholder={cell.kind === "markdown" ? "# markdown" : "# code"}
            onInlineEdit={model ? onInlineEdit : undefined}
          />
        </div>
      )}
      {outputs && <CellOutputList outputs={outputs} stale={stale} />}
    </div>
  )
}

export function NotebookView({ chatId, refreshKey, exists: existsHint, onOpenNotebookChanged }: NotebookViewProps) {
  const { selectedModel: model } = useSettings()

  const [mode, setMode] = useState<"edit" | "notebook">("notebook")
  const [source, setSource] = useState<string | null>(null)
  const [outputs, setOutputs] = useState<NotebookOutputs>({})
  const [revisionId, setRevisionId] = useState("")
  const [exists, setExists] = useState(!!existsHint)
  const [conflict, setConflict] = useState(false)
  const [loadError, setLoadError] = useState(false)
  const [saving, setSaving] = useState(false)
  const [runningCell, setRunningCell] = useState<number | null>(null)
  // Editing flags and the digest cache key by cell index: cell ids change on
  // every keystroke, and keying by them would close editors mid-edit.
  const [editingByIndex, setEditingByIndex] = useState<Record<number, boolean>>({})
  const [inlineEdit, setInlineEdit] = useState<InlineEditState | null>(null)
  const [digestByIndex, setDigestByIndex] = useState<Record<number, string>>({})

  const savedSourceRef = useRef("")
  const revisionRef = useRef(revisionId)
  revisionRef.current = revisionId
  // Last outputs each card showed live — kept so an edit dims them as stale
  // instead of dropping them the moment the digest stops matching.
  const rememberedOutputsRef = useRef<Record<number, NotebookCellOutput[]>>({})
  // True once a run response names this session's outputs authoritative.
  const hasRunRef = useRef(false)
  const saveTimer = useRef<number | undefined>(undefined)

  // Ref-stable callbacks so fetch effects and the debounce never re-run on identity.
  const existsChangedRef = useRef(onOpenNotebookChanged)
  existsChangedRef.current = onOpenNotebookChanged

  const load = useCallback(async () => {
    try {
      const snapshot = await fetchNotebook(chatId)
      setSource(snapshot.source)
      savedSourceRef.current = snapshot.source
      setOutputs(snapshot.outputs)
      setRevisionId(snapshot.revision_id)
      setConflict(false)
      setLoadError(false)
      setExists(true)
      existsChangedRef.current?.(true)
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        setExists(false)
        setLoadError(false)
        existsChangedRef.current?.(false)
        return
      }
      setLoadError(true)
    }
  }, [chatId])

  // Load on mount and whenever the activation hint turns on — the GET can race
  // the starter PUT and 404 first, so the hint's arrival must retry.
  useEffect(() => {
    setSource(null)
    setEditingByIndex({})
    setInlineEdit(null)
    load()
  }, [chatId, existsHint, refreshKey, load])

  const cells = useMemo(() => (source === null ? [] : parseNotebook(source)), [source])

  const persist = useCallback(async (next: string) => {
    setSaving(true)
    try {
      const { revision_id } = await saveNotebook(chatId, next, outputs, revisionRef.current)
      savedSourceRef.current = next
      setRevisionId(revision_id)
      setConflict(false)
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        // Another writer moved the revision: adopt its version, then say so.
        await load()
        setConflict(true)
      }
    } finally {
      setSaving(false)
    }
  }, [chatId, outputs, load])

  /** Debounce every save path through one timer; 409 inside persist reloads. */
  const scheduleSave = useCallback((serialized: string) => {
    clearTimeout(saveTimer.current)
    saveTimer.current = window.setTimeout(() => {
      if (serialized !== savedSourceRef.current) persist(serialized).catch(() => {})
    }, SAVE_DEBOUNCE_MS)
  }, [persist])

  const handleSourceChange = useCallback((next: string) => {
    setSource(next)
    scheduleSave(next)
  }, [scheduleSave])

  // Leaving the view flushes a pending save — the debounce cleanup alone would
  // drop edits typed in the final second.
  const flushRef = useRef<() => void>(() => {})
  flushRef.current = () => {
    if (source !== null && source !== savedSourceRef.current) persist(source).catch(() => {})
  }
  useEffect(() => () => {
    clearTimeout(saveTimer.current)
    flushRef.current()
  }, [])

  /** Replace one cell's source and schedule a save of the re-serialized file. */
  const replaceCellSource = useCallback((index: number, nextCellSource: string) => {
    setSource((current) => {
      if (current === null) return current
      const nextCells = parseNotebook(current)
      if (!nextCells[index]) return current
      nextCells[index] = { ...nextCells[index], source: nextCellSource }
      const serialized = serializeNotebook(nextCells)
      scheduleSave(serialized)
      return serialized
    })
  }, [scheduleSave])

  const toggleCellEditing = useCallback((index: number) => {
    setEditingByIndex((prev) => ({ ...prev, [index]: !prev[index] }))
  }, [])

  const runCell = useCallback(async (index: number) => {
    if (source === null || runningCell !== null) return
    clearTimeout(saveTimer.current)
    if (source !== savedSourceRef.current) await persist(source)
    setRunningCell(index)
    try {
      const result = await runNotebook(chatId, index + 1)
      hasRunRef.current = true
      setOutputs(result.outputs)
      setRevisionId(result.revision_id)
      setConflict(false)
    } catch (err) {
      // 409 means the revision moved; 404/422 are transient races (a cell
      // removed mid-click). Adopting the server's version covers all three.
      await load()
      if (err instanceof ApiError && err.status === 409) setConflict(true)
    } finally {
      setRunningCell(null)
    }
  }, [chatId, source, runningCell, load, persist])

  // Digest each code cell's source so its sidecar entry can be found.
  useEffect(() => {
    let cancelled = false
    Promise.all(
      cells.map(async (cell, index) => (cell.kind === "code" ? [index, await cellOutputKey(cell.source)] as const : null)),
    )
      .then((entries) => {
        if (cancelled) return
        setDigestByIndex((prev) => {
          const next = { ...prev }
          entries.forEach((entry) => { if (entry) next[entry[0]] = entry[1] })
          return next
        })
      })
      .catch(() => {})
    return () => { cancelled = true }
  }, [cells])

  // Remember the outputs each card shows live, keyed by the digest they belong
  // to; after an edit the digest misses and the remembered set renders dimmed
  // as stale. Once a re-run of the same digest reports no output, forget it.
  useEffect(() => {
    cells.forEach((cell, index) => {
      if (cell.kind !== "code") return
      const key = digestByIndex[index]
      if (!key) return
      if (outputs[key]) rememberedOutputsRef.current[index] = outputs[key]
      else if (hasRunRef.current) rememberedOutputsRef.current[index] = []
    })
  }, [cells, digestByIndex, outputs])

  const openInlineEdit = useCallback((cellIndex: number) => (text: string, start: number, end: number, splitPx: number) => {
    setInlineEdit({ cellIndex, text, start, end, splitPx })
  }, [])

  const applyInlineEdit = useCallback((replacement: string) => {
    if (!inlineEdit) return
    const { cellIndex, start, end } = inlineEdit
    setInlineEdit(null)
    if (cellIndex === -1) {
      // Whole-file Edit mode: splice straight into the serialized source.
      if (source === null) return
      handleSourceChange(source.substring(0, start) + replacement + source.substring(end))
      return
    }
    // Cell mode: splice into that cell's source, then re-serialize the file.
    const cell = cells[cellIndex]
    if (!cell) return
    replaceCellSource(cellIndex, cell.source.substring(0, start) + replacement + cell.source.substring(end))
  }, [inlineEdit, source, cells, handleSourceChange, replaceCellSource])

  if (!exists) return null
  if (source === null && loadError) {
    return (
      <div className="flex flex-col items-center gap-2 px-3 py-4 text-xs text-ink-faint">
        <span>Failed to load the notebook.</span>
        <button type="button" onClick={() => load()} className="rounded px-2 py-1 text-ink-subtle hover:text-ink hover:bg-hover transition-colors">
          Retry
        </button>
      </div>
    )
  }
  if (source === null) {
    return <div className="flex items-center justify-center px-3 py-4 text-xs text-ink-faint">Loading…</div>
  }

  // Whole-file inline-edit slices, mirroring the document editor's split.
  const fileSplitIdx = inlineEdit && inlineEdit.cellIndex === -1 ? source.indexOf("\n", inlineEdit.end) : -1
  const fileSplitChar = fileSplitIdx === -1 ? source.length : fileSplitIdx + 1
  const [fileTop, fileBot] = inlineEdit && inlineEdit.cellIndex === -1
    ? [source.substring(0, fileSplitChar), source.substring(fileSplitChar)]
    : ["", ""]

  return (
    <div className="flex flex-col">
      <div className="flex items-center justify-between gap-2 px-3 pb-2 pt-1">
        <NotebookModePills mode={mode} onModeChange={setMode} />
        <div className="flex items-center gap-1">
          {conflict && <span className="text-[10px] text-danger">changed elsewhere — showing the latest revision</span>}
          <button
            type="button"
            aria-label="Reload notebook"
            onClick={() => load()}
            className="rounded p-1 text-ink-subtle hover:text-ink hover:bg-surface-elevated transition-colors"
          >
            <RefreshCw className="h-3.5 w-3.5" />
          </button>
          <button
            type="button"
            aria-label="Save notebook"
            disabled={saving}
            onClick={() => {
              if (source !== savedSourceRef.current) persist(source).catch(() => {})
            }}
            className="rounded p-1 text-ink-subtle hover:text-ink hover:bg-surface-elevated disabled:opacity-30 transition-colors"
          >
            <Save className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>
      {mode === "edit" ? (
        inlineEdit && inlineEdit.cellIndex === -1 && model ? (
          <div className="flex flex-col border-t border-divider/40">
            <div style={{ height: inlineEdit.splitPx }} className="flex flex-col shrink-0 overflow-hidden">
              <ConsoleEditor value={fileTop} onChange={() => {}} language="python" readOnly />
            </div>
            <InlineEditPanel selectedText={inlineEdit.text} model={model} onApply={applyInlineEdit} onClose={() => setInlineEdit(null)} />
            <div style={{ height: editorHeight(fileBot) }} className="flex flex-col shrink-0">
              <ConsoleEditor value={fileBot} onChange={() => {}} language="python" readOnly startLineNumber={fileTop.split("\n").length} />
            </div>
          </div>
        ) : (
          <div className="flex h-[420px] flex-col border-t border-divider/40">
            <ConsoleEditor
              value={source}
              onChange={handleSourceChange}
              language="python"
              placeholder="# %% opens a code cell; # %% [markdown] opens a markdown cell"
              onInlineEdit={model ? openInlineEdit(-1) : undefined}
            />
          </div>
        )
      ) : (
        <div className="flex flex-col gap-2 px-3 pb-3">
          {cells.map((cell, index) => {
            const key = cell.kind === "code" ? digestByIndex[index] ?? "" : ""
            const live = key && outputs[key] ? outputs[key] : undefined
            const remembered = rememberedOutputsRef.current[index]
            const shown = live ?? (remembered && remembered.length > 0 ? remembered : null)
            const stale = shown !== null && live === undefined
            return (
              <NotebookCellCard
                key={index}
                cell={cell}
                index={index}
                editing={!!editingByIndex[index]}
                outputs={shown}
                stale={stale}
                onSourceChange={(next) => replaceCellSource(index, next)}
                onToggleEditing={() => toggleCellEditing(index)}
                onRun={() => runCell(index)}
                running={runningCell === index}
                onInlineEdit={openInlineEdit(index)}
                inlineEdit={inlineEdit && inlineEdit.cellIndex === index ? inlineEdit : null}
                onInlineApply={applyInlineEdit}
                onInlineClose={() => setInlineEdit(null)}
                model={model}
              />
            )
          })}
          {cells.length === 0 && (
            <div className="flex items-center justify-center gap-2 px-2 py-4 text-xs text-ink-faint">
              <BookOpen className="h-3.5 w-3.5" aria-hidden="true" />
              Empty notebook — start typing in Edit mode.
            </div>
          )}
        </div>
      )}
    </div>
  )
}