"use client"

import { useCallback, useState } from "react"
import { useRenameDocument } from "@/contexts/ProjectContext"
import { attachmentTraits } from "@/lib/domain/canvasAttachments"
import { A4_ASPECT, FALLBACK_ASPECT, omitKey, type AttachmentPatch, type CanvasAttachment, type CanvasDocument, type CanvasFrame } from "@/lib/domain/canvasDocument"

interface CanvasAttachmentsOptions {
  liveRef: { readonly current: { doc: CanvasDocument } }
  commit: (next: CanvasDocument) => void
  applyChange: (next: CanvasDocument) => void
  slug?: string
}

/** Frames and attachment windows once placed: measured aspects, removal, fullscreen, patches, retitles and the add menu's file renames and deletes. */
export function useCanvasAttachments({ liveRef, commit, applyChange, slug }: CanvasAttachmentsOptions) {
  const renameDocument = useRenameDocument()
  // Aspect cache (image frames + pdf attachments), keyed by element id
  const [aspects, setAspects] = useState<Record<string, number>>({})
  const setAspect = useCallback((id: string, r: number) => {
    setAspects((prev) => (Math.abs((prev[id] ?? 0) - r) < 0.001 ? prev : { ...prev, [id]: r }))
  }, [])
  const aspectOf = (id: string) => aspects[id] ?? FALLBACK_ASPECT
  const frameAspect = (f: CanvasFrame) => f.kind === "page" ? A4_ASPECT : aspectOf(f.id)

  /** Commits the document without entity `id` of `collection`, dropping its measured aspect. */
  const removeEntity = useCallback((collection: "frames" | "attachments", id: string) => {
    const cur = liveRef.current.doc
    commit({ ...cur, [collection]: cur[collection].filter((entity) => entity.id !== id) })
    setAspects((prev) => omitKey(prev, id))
  }, [liveRef, commit])
  const removeFrame = useCallback((id: string) => removeEntity("frames", id), [removeEntity])

  // Which attachment window, if any, is blown up over the whole surface. Exiting drops
  // straight back onto the canvas that holds it.
  const [fullscreenId, setFullscreenId] = useState<string | null>(null)
  const toggleFullscreen = useCallback((id: string) => setFullscreenId((current) => (current === id ? null : id)), [])

  const removeAttachment = useCallback((id: string) => {
    removeEntity("attachments", id)
    setFullscreenId((fullscreen) => (fullscreen === id ? null : fullscreen))
  }, [removeEntity])

  /** Applies `edit` to the live attachments; nothing is written when it hands them back unchanged. */
  const editAttachments = useCallback((edit: (all: CanvasAttachment[]) => CanvasAttachment[]) => {
    const cur = liveRef.current.doc
    const attachments = edit(cur.attachments)
    if (attachments !== cur.attachments) applyChange({ ...cur, attachments })
  }, [liveRef, applyChange])

  // Navigation, window state and titles (the tree's file, the PDF page, the connection style) persist without an undo step.
  // ponytail: nested canvas edits bypass the parent's history too — the nested editor has its own
  // undo/redo, and snapshotting the whole parent per nested stroke would be absurd.
  const patchAttachment = useCallback((id: string, patch: AttachmentPatch) => editAttachments((all) => {
    const attachment: Record<string, unknown> | undefined = all.find((a) => a.id === id)
    if (!attachment || Object.entries(patch).every(([key, value]) => attachment[key] === value)) return all
    return all.map((a) => a.id === id ? { ...a, ...patch } : a)
  }), [editAttachments])

  const retitleAttachment = useCallback(async (id: string, nextTitle: string) => {
    const attachment = liveRef.current.doc.attachments.find((a) => a.id === id)
    const title = nextTitle.trim()
    if (!attachment || !title || attachment.title === title) return
    if (!attachment.path || !attachmentTraits[attachment.kind].renamesFile) { patchAttachment(id, { title }); return }
    try {
      const renamed = await renameDocument(slug ?? "", attachment.path, title)
      patchAttachment(id, { path: renamed.path, title })
    } catch { /* the stored name stays authoritative when rename fails */ }
  }, [liveRef, patchAttachment, renameDocument, slug])

  // Attachments on this canvas follow a note the add menu renames (a project file keeps its key); they go once its delete commits, since the file is then gone for good.
  const followRenamedDocument = useCallback((oldPath: string, newPath: string) => {
    editAttachments((all) => all.some((a) => a.path === oldPath) ? all.map((a) => a.path === oldPath ? { ...a, path: newPath } : a) : all)
  }, [editAttachments])

  const dropDeletedDocument = useCallback((path: string) => {
    const gone = liveRef.current.doc.attachments.filter((a) => a.path === path).map((a) => a.id)
    if (gone.length === 0) return
    editAttachments((all) => all.filter((a) => a.path !== path))
    setFullscreenId((id) => (id && gone.includes(id) ? null : id))
  }, [liveRef, editAttachments])

  return {
    setAspect, aspectOf, frameAspect, removeFrame, removeAttachment, fullscreenId, toggleFullscreen,
    patchAttachment, retitleAttachment, followRenamedDocument, dropDeletedDocument,
  }
}
