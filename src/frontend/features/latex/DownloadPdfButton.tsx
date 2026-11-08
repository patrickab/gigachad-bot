"use client"

import { useState } from "react"
import { Download } from "lucide-react"
import { IconButton } from "@/components/ui/IconButton"
import { saveBlob } from "@/lib/download"

/** Saves the PDF at `pdfUrl` as `fileName`. Disabled without a PDF, red after a failed download until the next one works. */
export function DownloadPdfButton({ pdfUrl, fileName }: { pdfUrl: string | null; fileName: string }) {
  const [failed, setFailed] = useState(false)

  const download = async () => {
    if (!pdfUrl) return
    try {
      const response = await fetch(pdfUrl)
      if (!response.ok) throw new Error(response.statusText)
      saveBlob(await response.blob(), fileName)
      setFailed(false)
    } catch {
      setFailed(true)
    }
  }

  return <IconButton icon={Download} label="Download PDF" onClick={() => void download()} disabled={!pdfUrl} danger={failed} />
}
