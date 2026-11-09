"use client"

import dynamic from "next/dynamic"

// Own module, like LazyAnnotatedPdfViewer: a static import would pull React Flow, libavoid and the diagram surface into every importer's bundle.
export const LazyArchitectureWindow = dynamic(
  () => import("./ArchitectureWindow").then((m) => ({ default: m.ArchitectureWindow })),
  { ssr: false },
)
