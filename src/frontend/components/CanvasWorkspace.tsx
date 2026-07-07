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
}

const SCRATCH_STORAGE_KEY = "scratch-canvas-doc"

export function CanvasWorkspace({ selected, slug, toolbarSlot, onCloseEditor, onCreated, onModeLabel }: CanvasWorkspaceProps) {
  // The scratch canvas shown by default — no backing file until the user saves
  // it, so it lives in localStorage: every change persists, and leaving canvas
  // mode (which unmounts this component) loses nothing.
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

  useEffect(() => {
    if (!restoredRef.current) return
    try { localStorage.setItem(SCRATCH_STORAGE_KEY, serializeCanvasDoc(doc)) } catch { /* quota */ }
  }, [doc])

  // eslint-disable-next-line react-hooks/exhaustive-deps -- re-firing would only re-set the same label
  useEffect(() => {
    if (!selected) onModeLabel?.("untitled.canvas")
  }, [selected])

  useEffect(() => {
    if (selected) return
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "s") {
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
      setDoc(emptyCanvasDoc()) // content now lives in the named file — reset the scratch
      onCreated({ path: saved.path, scope })
    } catch { /* */ }
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
      />
    )
  }

  return (
    <div className="relative h-full">
      <CanvasEditor doc={doc} onChange={setDoc} toolbarSlot={toolbarSlot} />
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
