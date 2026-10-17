"use client"

import dynamic from "next/dynamic"

// The single lazy entry point into PdfViewer. It lives in its own module on
// purpose: a dynamic import is only code-split and client-only while nothing
// static pulls the target module in, so re-exporting this from PdfViewer.tsx
// itself would drag react-pdf and pdf.js into every importer's server bundle.
export const LazyPdfViewer = dynamic(
  () => import("./PdfViewer").then((m) => ({ default: m.PdfViewer })),
  { ssr: false },
)
