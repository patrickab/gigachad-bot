"use client"

import { useCallback, useEffect } from "react"
import { A4_WIDTH, CANVAS_ASPECT, FALLBACK_ASPECT, canvasEntityId, emptyCanvasDoc, type CanvasAttachment, type CanvasDocument } from "@/lib/domain/canvasDocument"
import { writeBinaryDocument } from "@/lib/backend/api"

const PAGE_GAP = 40
const DEFAULT_PDF_WIDTH = 500 // attachment default width (canvas units)
const DEFAULT_IMAGE_WIDTH = 400 // image-frame default width (canvas units)
const DEFAULT_CANVAS_WIDTH = 600 // nested canvas window default width (canvas units)
const DEFAULT_GRAPH_WIDTH = 720
const DEFAULT_GRAPH_HEIGHT = 480

type NewAttachment = Pick<CanvasAttachment, "kind" | "path" | "canvas" | "height">

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

  // One attachment window centered in the view. `dedupe` skips a path that is already on the canvas.
  const insertAttachment = useCallback((fields: NewAttachment, width: number, aspect: number, dedupe = false) => {
    const doc = liveRef.current.doc
    if (dedupe && doc.attachments.some((a) => a.path === fields.path)) return
    const { cx, cy } = pointAtCenter(width, aspect)
    commit({ ...doc, attachments: [...doc.attachments, { id: canvasEntityId(), ...fields, x: cx, y: cy, width }] })
  }, [liveRef, commit, pointAtCenter])

  const addPdf = useCallback((path: string) => insertAttachment({ kind: "pdf", path }, DEFAULT_PDF_WIDTH, FALLBACK_ASPECT, true), [insertAttachment])

  // Nested canvas window — a full CanvasEditor living inside this one, contents stored inline in the parent document.
  const addNewCanvas = useCallback(() => insertAttachment({ kind: "canvas", canvas: emptyCanvasDoc() }, DEFAULT_CANVAS_WIDTH, CANVAS_ASPECT), [insertAttachment])

  // A referenced canvas stores only its placement here; its document is owned by its own collaboration hook.
  const addCanvasFile = useCallback((path: string) => insertAttachment({ kind: "canvas", path }, DEFAULT_CANVAS_WIDTH, CANVAS_ASPECT, true), [insertAttachment])

  // A project has one architecture; each window browses it through its own tree, opening on the
  // first system. Several windows on one project are intended, so this never dedupes.
  const addArchitecture = useCallback(() => {
    if (!slug) return
    insertAttachment({ kind: "architecture-graph", path: `graph/${slug}/`, height: DEFAULT_GRAPH_HEIGHT }, DEFAULT_GRAPH_WIDTH, DEFAULT_GRAPH_HEIGHT / DEFAULT_GRAPH_WIDTH)
  }, [insertAttachment, slug])

  const addDocument = useCallback((path: string) => insertAttachment({ kind: "document", path, height: DEFAULT_GRAPH_HEIGHT }, DEFAULT_GRAPH_WIDTH, DEFAULT_GRAPH_HEIGHT / DEFAULT_GRAPH_WIDTH, true), [insertAttachment])

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
  useEffect(() => {
    const handler = (e: ClipboardEvent) => {
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
    }
    window.addEventListener("paste", handler)
    return () => window.removeEventListener("paste", handler)
  }, [active, slug, depth, pointAtCenter, uploadPastedImage])

  return { addPage, addImage, addPdf, addNewCanvas, addCanvasFile, addArchitecture, addDocument, pasteImageAtPoint }
}
