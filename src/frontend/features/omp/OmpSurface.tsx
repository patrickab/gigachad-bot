"use client"

import { useEffect, useEffectEvent, useState } from "react"
import { useTab } from "@/contexts/TabContext"
import { useOpenFile } from "@/features/editor/FilePanel"
import type { OmpView } from "@/lib/domain/types"
import { useOmp } from "./OmpProvider"
import { OmpComposer } from "./OmpComposer"
import { viewTitle } from "./sessions"

/** One omp tab: the slot the shared frame is laid over while the tab is frontmost, with gigachad's composer below it. */
export function OmpSurface({ view, onViewChange }: { view: OmpView; onViewChange: (view: OmpView) => void }) {
  const { tab, isActive, setTitle } = useTab()
  const { attachFrame, frameStatus, ompWebUrl, overview, reloadFrame } = useOmp()
  const openFile = useOpenFile()
  const [slot, setSlot] = useState<HTMLDivElement | null>(null)
  const title = viewTitle(view, overview)
  const adoptView = useEffectEvent((next: OmpView) => onViewChange(next))

  useEffect(() => {
    if (!isActive || !slot) return
    return attachFrame({ slot, view, onViewChange: (next) => adoptView(next), onOpenFile: openFile })
  }, [isActive, slot, view, attachFrame, openFile])

  useEffect(() => {
    setTitle(title)
  }, [title, setTitle])

  return (
    <div className="flex h-full w-full flex-col">
      <div ref={setSlot} className="flex min-h-0 w-full flex-1 items-center justify-center">
        {frameStatus === "unreachable" ? (
          <div className="flex flex-col items-center gap-2 text-xs text-ink-subtle">
            <p>omp-web is not reachable at {ompWebUrl}</p>
            <button type="button" onClick={reloadFrame} className="rounded px-2 py-1 text-ink-muted hover:bg-hover hover:text-ink">Retry</button>
          </div>
        ) : frameStatus === "loading" && (
          <p className="text-xs text-ink-subtle">Loading omp…</p>
        )}
      </div>
      {view.kind !== "settings" && <OmpComposer view={view} chatId={tab.chatId} slug={tab.projectSlug} />}
    </div>
  )
}
