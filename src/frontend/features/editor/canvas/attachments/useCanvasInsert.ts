"use client"

import { useCallback } from "react"
import { useEventListener } from "@/hooks/useEventListener"
import { attachmentTraits } from "@/lib/domain/canvasAttachments"
import { A4_WIDTH, FALLBACK_ASPECT, canvasEntityId, emptyCanvasDoc, type CanvasAttachment, type CanvasDocument } from "@/lib/domain/canvasDocument"
import { writeBinaryDocument } from "@/lib/backend/api"

const PAGE_GAP = 40
const DEFAULT_IMAGE_WIDTH = 400 // image-frame default width (canvas units)

/** A new attachment's own fields, per kind; placement and size come from its kind's traits. */
type OwnFields<A> = A extends CanvasAttachment ? Omit<A, "id" | "x" | "y" | "width" | "height"> : never
type NewAttachment = OwnFields<CanvasAttachment>

interface CanvasInsertOptions {
  liveRef: { readonly current: { doc: CanvasDocument } }
  commit: (next: CanvasDocument) => void
  pointAtCenter: (width: number, aspect: number) => { cx: number; cy: number }
  slug?: string
  depth: number
  /** Gates the window paste listener to the frontmost tab. */
  active: boolean
  onImageAdded?: (path: string) => void
}

/** Everything that puts new content on a canvas: pages, image frames, attachment windows and pasted images. */
export function useCanvasInsert({ liveRef, commit, pointAtCenter, slug, depth, active, onImageAdded }: CanvasInsertOptions) {
  const addPage = useCallback(() => {
    const doc = liveRef.current.doc
    const pages = doc.frames.filter((f) => f.kind === "page")
    const last = pages[pages.length - 1]
    const newX = last ? last.x + last.width + PAGE_GAP : 0
    commit({ ...doc, frames: [...doc.frames, { id: canvasEntityId(), kind: "page", x: newX, y: last?.y ?? 0, width: A4_WIDTH }] })
  }, [liveRef, commit])

  const addImageFrameAt = useCallback((path: string, x: number, y: number) => {
    const doc = liveRef.current.doc
    commit({ ...doc, frames: [...doc.frames, { id: canvasEntityId(), kind: "image", x, y, width: DEFAULT_IMAGE_WIDTH, path }] })
  }, [liveRef, commit])

  const addImage = useCallback((path: string) => {
    const { cx, cy } = pointAtCenter(DEFAULT_IMAGE_WIDTH, FALLBACK_ASPECT)
    addImageFrameAt(path, cx, cy)
  }, [pointAtCenter, addImageFrameAt])

  // One attachment window of its kind's default size, centered in the view. `dedupe` skips a path that is already on the canvas.
  const insertAttachment = useCallback((fields: NewAttachment, dedupe = false) => {
    const doc = liveRef.current.doc
    if (dedupe && doc.attachments.some((a) => a.path === fields.path)) return
    const size = attachmentTraits[fields.kind].defaultSize
    const { cx, cy } = pointAtCenter(size.width, "height" in size ? size.height / size.width : size.aspect)
    const attachment: CanvasAttachment = { id: canvasEntityId(), ...fields, ...("height" in size && { height: size.height }), x: cx, y: cy, width: size.width }
    commit({ ...doc, attachments: [...doc.attachments, attachment] })
  }, [liveRef, commit, pointAtCenter])

  const addPdf = useCallback((path: string) => insertAttachment({ kind: "pdf", path }, true), [insertAttachment])

  // Nested canvas window — a full CanvasEditor living inside this one, contents stored inline in the parent document.
  const addNewCanvas = useCallback(() => insertAttachment({ kind: "canvas", canvas: emptyCanvasDoc() }), [insertAttachment])

  // A referenced canvas stores only its placement here; its document is owned by its own collaboration hook.
  const addCanvasFile = useCallback((path: string) => insertAttachment({ kind: "canvas", path }, true), [insertAttachment])

  // A project has one architecture; each window browses it through its own tree, opening on the
  // first system. Several windows on one project are intended, so this never dedupes.
  const addArchitecture = useCallback(() => {
    if (!slug) return
    insertAttachment({ kind: "architecture-graph", path: `graph/${slug}/` })
  }, [insertAttachment, slug])

  const addDocument = useCallback((path: string) => insertAttachment({ kind: "document", path }, true), [insertAttachment])

  // Store a clipboard image in the project and drop it on the canvas at (x, y).
  const uploadPastedImage = useCallback(async (blob: Blob, mime: string, x: number, y: number) => {
    if (!slug) return
    const name = `pasted-${Date.now()}.${mime.split("/")[1] || "png"}`
    const res = await writeBinaryDocument(slug, name, blob)
    addImageFrameAt(res.path, x, y)
    onImageAdded?.(res.path)
  }, [slug, addImageFrameAt, onImageAdded])

  // Paste via the long-press menu: read the image off the async clipboard.
  const pasteImageAtPoint = useCallback(async (canvasX: number, canvasY: number) => {
    if (!slug) return
    try {
      for (const item of await navigator.clipboard.read()) {
        const imageType = item.types.find((t) => t.startsWith("image/"))
        if (!imageType) continue
        await uploadPastedImage(await item.getType(imageType), imageType, canvasX, canvasY)
        return
      }
    } catch { /* clipboard empty or no permission */ }
  }, [slug, uploadPastedImage])

  // Ctrl+V
  useEventListener("window", "paste", (e) => {
    // every canvas window listens on `window`; only the outermost, active one may act
    if (!active || !slug || depth > 0) return
    for (const item of e.clipboardData?.items ?? []) {
      if (!item.type.startsWith("image/")) continue
      e.preventDefault()
      const blob = item.getAsFile()
      if (!blob) continue
      const { cx, cy } = pointAtCenter(DEFAULT_IMAGE_WIDTH, FALLBACK_ASPECT)
      uploadPastedImage(blob, item.type, cx, cy).catch(() => {})
      return
    }
  })

  return { addPage, addImage, addPdf, addNewCanvas, addCanvasFile, addArchitecture, addDocument, pasteImageAtPoint }
}
