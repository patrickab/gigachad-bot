"use client"

import dynamic from "next/dynamic"

// Own module, like LazyPdfViewer: a static import would pull react-pdf and pdf.js into every importer's bundle.
export const LazyAnnotatedPdfViewer = dynamic(
  () => import("./AnnotatedPdfViewer").then((m) => ({ default: m.AnnotatedPdfViewer })),
  { ssr: false },
)
