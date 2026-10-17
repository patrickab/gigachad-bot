"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { PenLine } from "lucide-react"
import { CanvasEditor, emptyCanvasDoc, parseCanvasDoc, serializeCanvasDoc, type CanvasDocument } from "./CanvasEditor"
import { DocumentEditor } from "./DocumentEditor"
import { SaveChatModal } from "./SaveChatModal"
import { writeDocument } from "@/lib/api"

// A canvas selection carries its owning scope: "" = chat_histories/_notes, otherwise a project slug.
export interface CanvasSelection {
  path: string
  scope: string
}

interface CanvasWorkspaceProps {
  selected: CanvasSelection | null
  slug: string | null
  toolbarSlot: HTMLElement | null
  onCloseEditor: () => void
  onCreated: (sel: CanvasSelection) => void
  onModeLabel?: (label: string) => void
  /** Model behind the document editor's LLM features (assistant sidebar, inline edit). */
  editorModel?: string
}

const SCRATCH_STORAGE_KEY = "scratch-canvas-doc"

export function CanvasWorkspace({ selected, slug, toolbarSlot, onCloseEditor, onCreated, onModeLabel, editorModel }: CanvasWorkspaceProps) {
  // Scratch canvas shown by default. It has no backing file until saved, so it
  // persists to localStorage on every change and survives unmounting.
  const [doc, setDoc] = useState<CanvasDocument>(() => emptyCanvasDoc())
  const [saveOpen, setSaveOpen] = useState(false)

  const restoredRef = useRef(false)
  useEffect(() => {
    if (restoredRef.current) return
    restoredRef.current = true
    try {
      const s = localStorage.getItem(SCRATCH_STORAGE_KEY)
      if (s?.trim()) setDoc(parseCanvasDoc(s))
    } catch { /* corrupt scratch — start empty */ }
  }, [])

  // Debounced: serializing the doc and writing it is O(whole canvas) and localStorage is
  // synchronous, so doing it per stroke/pointermove stalls the drawing surface itself.
  useEffect(() => {
    if (!restoredRef.current) return
    const t = setTimeout(() => {
      try { localStorage.setItem(SCRATCH_STORAGE_KEY, serializeCanvasDoc(doc)) } catch { /* quota */ }
    }, 500)
    return () => clearTimeout(t)
  }, [doc])

  // Unmount skips the pending debounce, so flush the latest doc here.
  const docRef = useRef(doc)
  docRef.current = doc
  useEffect(() => () => {
    try { localStorage.setItem(SCRATCH_STORAGE_KEY, serializeCanvasDoc(docRef.current)) } catch { /* quota */ }
  }, [])

  // eslint-disable-next-line react-hooks/exhaustive-deps -- re-firing would only re-set the same label
  useEffect(() => {
    if (!selected) onModeLabel?.("untitled.canvas")
  }, [selected])

  useEffect(() => {
    if (selected) return
    // Alt+S opens the save-as dialog for the scratch canvas
    const handler = (e: KeyboardEvent) => {
      if (e.altKey && e.key === "s") {
        e.preventDefault()
        setSaveOpen(true)
      }
    }
    document.addEventListener("keydown", handler)
    return () => document.removeEventListener("keydown", handler)
  }, [selected])

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
        canvasToolbarSlot={toolbarSlot}
        onClose={onCloseEditor}
        onModeLabel={onModeLabel}
        model={editorModel}
      />
    )
  }

  return (
    <div className="relative h-full">
      {/* The scratch canvas belongs to the open project (it is saved there too), so it offers that project's documents and architecture. */}
      <CanvasEditor doc={doc} onChange={setDoc} slug={slug ?? undefined} toolbarSlot={toolbarSlot} />
      <SaveChatModal
        open={saveOpen}
        onClose={() => setSaveOpen(false)}
        onSave={handleSaveSubmit}
        title="Save Canvas"
        icon={PenLine}
      />
    </div>
  )
}
