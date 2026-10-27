import { LazyPdfViewer } from "@/components/content/LazyPdfViewer"
import { fileViewerRawUrl } from "@/lib/backend/urls"
import { baseName } from "@/lib/paths"

/** Image and PDF render from the raw file URL, the way the file finder previews them. */
export function RawFileView({ path, kind }: { path: string; kind: "image" | "pdf" }) {
  if (kind === "pdf") return <div className="min-h-0 flex-1"><LazyPdfViewer key={path} url={fileViewerRawUrl(path)} /></div>
  return (
    <div className="min-h-0 flex-1 overflow-y-auto p-4">
      {/* eslint-disable-next-line @next/next/no-img-element -- dynamic backend raw-file URL in a Tauri static export, next/image cannot optimize it */}
      <img src={fileViewerRawUrl(path)} alt={baseName(path)} className="mx-auto max-h-full max-w-full rounded border border-divider object-contain" />
    </div>
  )
}
