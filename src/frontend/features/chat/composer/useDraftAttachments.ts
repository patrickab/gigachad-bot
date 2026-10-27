"use client"

import { useCallback, useState } from "react"
import { uploadFile } from "@/lib/backend/api"
import { attachReference } from "@/lib/backend/references"
import { readReferences, saveAttachmentEdit, withReferenceRead } from "@/features/chat/attachments"
import type { Attachment } from "@/lib/domain/types"

/** The composer's draft attachments: uploads in flight, intake, removal, and the full preview of one of them. */
export function useDraftAttachments(chatId: string, slug: string | null) {
  const [attachments, setAttachments] = useState<Attachment[]>([])
  const [uploadingNames, setUploadingNames] = useState<ReadonlySet<string>>(new Set())
  // Name of the draft attachment whose full preview is open. Earlier messages' chips open the chat's preview.
  const [previewName, setPreviewName] = useState<string | null>(null)

  const replaceNamed = (name: string, update: (attachment: Attachment) => Attachment) =>
    setAttachments((prev) => prev.map((a) => a.name === name ? update(a) : a))

  /** Add `attachment`, replacing a draft attachment of the same name. */
  const add = useCallback((attachment: Attachment) => setAttachments((prev) => {
    const existing = prev.findIndex((a) => a.name === attachment.name)
    if (existing < 0) return [...prev, attachment]
    const copy = [...prev]
    copy[existing] = attachment
    return copy
  }), [])
  const remove = useCallback((name: string) => setAttachments((prev) => prev.filter((a) => a.name !== name)), [])
  const clear = useCallback(() => setAttachments([]), [])

  /** Upload `file` and append it; `name` marks its chip as uploading meanwhile. A failed upload attaches nothing. */
  const upload = useCallback((file: File, name: string) => {
    setUploadingNames((prev) => new Set(prev).add(name))
    uploadFile(chatId, file, slug)
      .then((att) => setAttachments((prev) => [...prev, att]))
      .catch(() => {})
      .finally(() => setUploadingNames((prev) => { const next = new Set(prev); next.delete(name); return next }))
  }, [chatId, slug])

  const edit = async (name: string, content: string) => {
    const attachment = attachments.find((a) => a.name === name)
    if (!attachment) return
    const saved = await saveAttachmentEdit(attachment, content)
    replaceNamed(name, () => saved)
  }

  /** Open a draft attachment's preview, a reference re-read first so Neovim never edits a stale copy. */
  const openPreview = async (attachment: Attachment) => {
    if (attachment.vaultPath) {
      const reads = await readReferences([attachment.vaultPath], attachReference)
      replaceNamed(attachment.name, (a) => withReferenceRead(a, reads))
    }
    setPreviewName(attachment.name)
  }

  return {
    attachments,
    uploadingNames,
    add,
    remove,
    clear,
    upload,
    // Resolved each render so the window follows edits and closes once its attachment is removed.
    preview: attachments.find((a) => a.name === previewName) ?? null,
    openPreview,
    closePreview: () => setPreviewName(null),
    /** Save an edit of the previewed attachment, undefined while none is open. */
    editPreview: previewName == null ? undefined : (content: string) => edit(previewName, content),
  }
}
