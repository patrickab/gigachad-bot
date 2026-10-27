"use client"

import { useCallback, useEffect, useEffectEvent, useState } from "react"
import { PenLine } from "lucide-react"
import { CanvasEditor } from "./CanvasEditor"
import { emptyCanvasDoc, parseCanvasDoc, serializeCanvasDoc, type CanvasDocument } from "@/lib/domain/canvasDocument"
import { DocumentEditor } from "@/features/editor/DocumentEditor"
import { SaveAsModal } from "@/components/ui/SaveAsModal"
import { writeDocument } from "@/lib/backend/api"
import { useTabActive } from "@/contexts/TabContext"
import { useEventListener } from "@/hooks/useEventListener"
import { useModeLabel } from "@/hooks/useModeLabel"
import { readStorage, writeStorage } from "@/lib/storage"

// A canvas selection carries its owning scope: "" = chat_histories/_notes, otherwise a project slug.
export interface CanvasSelection {
  path: string
  scope: string
}

interface CanvasWorkspaceProps {
  selected: CanvasSelection | null
  slug: string | null
  onCloseEditor: () => void
  onCreated: (sel: CanvasSelection) => void
  onModeLabel?: (label: string) => void
  /** Model behind the document editor's LLM features (assistant sidebar, inline edit). */
  editorModel?: string
}

const SCRATCH_STORAGE_KEY = "scratch-canvas-doc"

/** The scratch canvas persisted by a previous session, or an empty one when absent or corrupt. */
function loadScratchDoc(): CanvasDocument {
  try {
    const stored = readStorage(SCRATCH_STORAGE_KEY)
    if (stored?.trim()) return parseCanvasDoc(stored)
  } catch { /* corrupt scratch — start empty */ }
  return emptyCanvasDoc()
}

export function CanvasWorkspace({ selected, slug, onCloseEditor, onCreated, onModeLabel, editorModel }: CanvasWorkspaceProps) {
  const isActive = useTabActive()
  // Scratch canvas shown by default. It has no backing file until saved, so it
  // persists to localStorage on every change and survives unmounting.
  const [doc, setDoc] = useState<CanvasDocument>(loadScratchDoc)
  const [saveOpen, setSaveOpen] = useState(false)

  // Debounced: serializing the doc and writing it is O(whole canvas) and localStorage is
  // synchronous, so doing it per stroke/pointermove stalls the drawing surface itself.
  const saveScratch = useEffectEvent(() => writeStorage(SCRATCH_STORAGE_KEY, serializeCanvasDoc(doc)))
  useEffect(() => {
    const t = setTimeout(() => saveScratch(), 500)
    return () => clearTimeout(t)
  }, [doc])

  // Unmount skips the pending debounce, so flush the latest doc here.
  useEffect(() => () => { saveScratch() }, [])

  useModeLabel("new.canvas", !selected, onModeLabel)

  // Alt+S opens the save-as dialog for the scratch canvas
  useEventListener("document", "keydown", (e) => {
    if (e.altKey && e.key === "s") {
      e.preventDefault()
      setSaveOpen(true)
    }
  }, { active: !selected && isActive })

  const handleSaveSubmit = useCallback(async (name: string) => {
    const filename = name.endsWith(".canvas") ? name : `${name}.canvas`
    const scope = slug ?? ""
    try {
      const saved = await writeDocument(scope, filename, serializeCanvasDoc(doc))
      setSaveOpen(false)
      setDoc(emptyCanvasDoc()) // content now lives in the named file, reset the scratch
      onCreated({ path: saved.path, scope })
    } catch { /* keep the dialog open so the user can retry */ }
  }, [slug, doc, onCreated])

  if (selected) {
    return (
      <DocumentEditor
        key={selected.path}
        path={selected.path}
        slug={selected.scope}
        overlay
        onClose={onCloseEditor}
        onModeLabel={onModeLabel}
        model={editorModel}
      />
    )
  }

  return (
    <div className="relative h-full">
      {/* The scratch canvas belongs to the open project (it is saved there too), so it offers that project's documents and architecture. */}
      <CanvasEditor doc={doc} onChange={setDoc} slug={slug ?? undefined} />
      <SaveAsModal
        open={saveOpen}
        onClose={() => setSaveOpen(false)}
        onSave={handleSaveSubmit}
        title="Save Canvas"
        icon={PenLine}
      />
    </div>
  )
}
