"use client"

import { useState } from "react"
import { Presentation } from "lucide-react"
import { IconButton } from "@/components/ui/IconButton"
import { openPresentWindow } from "./presentParams"

/** Opens the presenter window on a fresh channel, the presenter opens the audience window from there. Turns red when the browser blocks the popup. */
export function PresentButton({ pdfUrl }: { pdfUrl: string | null }) {
  const [blocked, setBlocked] = useState(false)

  const present = () => {
    if (!pdfUrl) return
    // Not crypto.randomUUID: it only exists in secure contexts, and a BroadcastChannel never leaves this origin anyway
    const channel = Math.random().toString(36).slice(2)
    setBlocked(!openPresentWindow({ pdfUrl, role: "presenter", channel }))
  }

  return <IconButton icon={Presentation} label="Present" onClick={present} disabled={!pdfUrl} danger={blocked} />
}
