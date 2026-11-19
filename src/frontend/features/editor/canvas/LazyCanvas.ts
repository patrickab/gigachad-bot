"use client"

import { createContext } from "react"
import dynamic from "next/dynamic"

/** Receives every canvas screenshot as a PNG, beside the clipboard copy, e.g. to attach it to the canvas's chat. Here, not beside the screenshot code, so its provider stays out of the canvas bundle. */
export const CanvasCaptureContext = createContext<((png: File) => void) | null>(null)

// Own module, like LazyAnnotatedPdfViewer: a static import would pull the canvas editor (ink, camera, attachments, mutation sync) into every importer's bundle.
export const LazyCanvasWorkspace = dynamic(
  () => import("./CanvasWorkspace").then((m) => ({ default: m.CanvasWorkspace })),
  { ssr: false },
)

export const LazyCanvasDocumentEditor = dynamic(
  () => import("./CanvasDocumentEditor").then((m) => ({ default: m.CanvasDocumentEditor })),
  { ssr: false },
)

export const LazyFileCanvasEditor = dynamic(
  () => import("./FileCanvasEditor").then((m) => ({ default: m.FileCanvasEditor })),
  { ssr: false },
)
