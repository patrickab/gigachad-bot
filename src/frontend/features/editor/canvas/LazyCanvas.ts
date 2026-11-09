"use client"

import dynamic from "next/dynamic"

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
