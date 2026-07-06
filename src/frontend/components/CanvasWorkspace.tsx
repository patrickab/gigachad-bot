"use client"

import { useCallback, useEffect, useState } from "react"
import { PenLine } from "lucide-react"
import { CanvasEditor, emptyCanvasDoc, serializeCanvasDoc, type CanvasDocument } from "./CanvasEditor"
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

export function CanvasWorkspace({ selected, slug, toolbarSlot, onCloseEditor, onCreated, onModeLabel }: CanvasWorkspaceProps) {
  // The scratch canvas shown by default — not persisted until the user saves it.
  const [doc, setDoc] = useState<CanvasDocument>(() => emptyCanvasDoc())
  const [saveOpen, setSaveOpen] = useState(false)

  // eslint-disable-next-line react-hooks/exhaustive-deps -- re-firing would only re-set the same label
  useEffect(() => {
    if (!selected) {
      setDoc(emptyCanvasDoc())
      onModeLabel?.("untitled.canvas")
    }
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
