"use client"

import { useEffect, useEffectEvent, useState } from "react"
import { useTab } from "@/contexts/TabContext"
import type { OmpView } from "@/lib/domain/types"
import { useOmp } from "./OmpProvider"
import { viewTitle } from "./sessions"

/** One omp tab's slot: the shared frame is laid over it while the tab is frontmost. */
export function OmpSurface({ view, onViewChange }: { view: OmpView; onViewChange: (view: OmpView) => void }) {
  const { isActive, setTitle } = useTab()
  const { attachFrame, frameStatus, ompWebUrl, overview, reloadFrame } = useOmp()
  const [slot, setSlot] = useState<HTMLDivElement | null>(null)
  const title = viewTitle(view, overview)
  const adoptView = useEffectEvent((next: OmpView) => onViewChange(next))

  useEffect(() => {
    if (!isActive || !slot) return
    return attachFrame({ slot, view, onViewChange: (next) => adoptView(next) })
  }, [isActive, slot, view, attachFrame])

  useEffect(() => {
    setTitle(title)
  }, [title, setTitle])

  return (
    <div ref={setSlot} className="flex h-full w-full items-center justify-center">
      {frameStatus === "unreachable" ? (
        <div className="flex flex-col items-center gap-2 text-xs text-ink-subtle">
          <p>omp-web is not reachable at {ompWebUrl}</p>
          <button type="button" onClick={reloadFrame} className="rounded px-2 py-1 text-ink-muted hover:bg-hover hover:text-ink">Retry</button>
        </div>
      ) : frameStatus === "loading" && (
        <p className="text-xs text-ink-subtle">Loading omp…</p>
      )}
    </div>
  )
}
