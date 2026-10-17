"use client"

import { memo } from "react"
import { Download, FileText } from "lucide-react"
import { sandboxAssetUrl } from "@/lib/backend/urls"
import type { PlotFigure, SandboxOutputRecord } from "@/lib/domain/types"
import { PlotElement } from "@/components/content/PlotElement"

/** Vendor MIME type, since Plotly registers none. */
const PLOTLY_MEDIA_TYPE = "application/vnd.plotly.v1+json"

interface SandboxOutputElementProps {
  output: SandboxOutputRecord
}

/** Parse the figure JSON, or null when it is malformed so the caller falls back to other renderings. */
function parsePlotFigure(text: string): PlotFigure | null {
  try {
    return JSON.parse(text) as PlotFigure
  } catch {
    return null
  }
}

function SandboxOutputElementInner({ output }: SandboxOutputElementProps) {
  const mediaTypes = Object.keys(output.mime_bundle)

  if (mediaTypes.includes(PLOTLY_MEDIA_TYPE) && output.text) {
    const figure = parsePlotFigure(output.text)
    if (figure) return <PlotElement figure={figure} />
  }

  const image = mediaTypes.find((t) => t.startsWith("image/"))
  if (image) {
    const ref = output.mime_bundle[image]
    // Sandbox assets are served by the backend at runtime, so next/image optimization does not apply.
    // eslint-disable-next-line @next/next/no-img-element -- dynamic backend asset URL under Tauri static export
    return <img src={sandboxAssetUrl(ref.asset_path)} alt={output.title ?? "Sandbox output"} className="max-w-full rounded-lg" />
  }

  if (mediaTypes.includes("application/pdf")) {
    const ref = output.mime_bundle["application/pdf"]
    return (
      <a
        href={sandboxAssetUrl(ref.asset_path)}
        target="_blank"
        rel="noopener noreferrer"
        className="flex items-center gap-2 text-sm text-ink underline decoration-divider-strong"
      >
        <FileText className="h-3.5 w-3.5" aria-hidden="true" />
        {output.title ?? "Open PDF"}
      </a>
    )
  }

  if (output.text && (mediaTypes.includes("text/markdown") || mediaTypes.includes("text/plain"))) {
    return <p className="whitespace-pre-wrap text-sm text-ink-muted">{output.text}</p>
  }

  const first = mediaTypes[0]
  const ref = first ? output.mime_bundle[first] : null
  if (!ref) return null
  return (
    <a
      href={sandboxAssetUrl(ref.asset_path)}
      target="_blank"
      rel="noopener noreferrer"
      className="flex items-center gap-2 text-[10px] text-ink-faint underline decoration-divider-strong"
    >
      <Download className="h-3 w-3" aria-hidden="true" />
      {output.title ?? first} · {ref.size_bytes} bytes
    </a>
  )
}

export const SandboxOutputElement = memo(SandboxOutputElementInner)