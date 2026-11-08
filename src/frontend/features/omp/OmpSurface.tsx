"use client"

import { useEffect, useEffectEvent, useRef, useState } from "react"
import { useIsPeeked, useTab } from "@/contexts/TabContext"
import { useOpenFile } from "@/features/editor/FilePanel"
import { useElementSize } from "@/hooks/useElementSize"
import type { OmpView } from "@/lib/domain/types"
import type { FrameBackdrop } from "@gigachad/chat-ui/protocol"
import { useOmp } from "./OmpProvider"
import { OmpComposer, type ComposerInsertion, type ShellCommands } from "./OmpComposer"
import { viewTitle } from "./sessions"

/** Height of the shell's top bar (`TopBar`, h-11) the transcript scrolls beneath. */
const TOP_BAR_PX = 44

/**
 * One omp tab: the slot the shared frame is laid over while the tab is frontmost or previewed in the tab picker.
 *
 * - Sessions run under the top bar and gigachad's composer like a chat, the frame pads its transcript by these insets
 * - Settings keep the plain layout below the top bar (`WorkspaceSurface`)
 * - A drawer inside another surface passes `topInset` 0 and `adoptTitle` false, so the tab keeps its own title
 * - `backdrop` names the shell surface behind it, the frame and the composer's fade take its color
 * - `insertion` goes into the composer draft once (`OmpComposer`), `lead` leads the next message, `onLeadSent` follows it
 */
export function OmpSurface({ view, onViewChange, shellCommands, topInset = TOP_BAR_PX, adoptTitle = true, backdrop = "paper", insertion, onInserted, lead, onLeadSent }: {
  view: OmpView
  onViewChange: (view: OmpView) => void
  shellCommands: ShellCommands
  topInset?: number
  adoptTitle?: boolean
  backdrop?: FrameBackdrop
  insertion?: ComposerInsertion | null
  onInserted?: () => void
  lead?: string
  onLeadSent?: () => void
}) {
  const { tab, isActive, setTitle } = useTab()
  const peeked = useIsPeeked()
  const { attachFrame, framePeeked, setFrameInsets, frameStatus, ompWebUrl, overview, reloadFrame } = useOmp()
  const openFile = useOpenFile()
  const [slot, setSlot] = useState<HTMLDivElement | null>(null)
  const title = viewTitle(view, overview)
  const adoptView = useEffectEvent((next: OmpView) => onViewChange(next))

  useEffect(() => {
    if (!(isActive || peeked) || !slot) return
    return attachFrame({ slot, view, onViewChange: (next) => adoptView(next), onOpenFile: openFile, peek: peeked })
  }, [isActive, peeked, slot, view, attachFrame, openFile])

  useEffect(() => {
    if (adoptTitle) setTitle(title)
  }, [adoptTitle, title, setTitle])

  const hasComposer = view.kind !== "settings"
  const composerRef = useRef<HTMLDivElement>(null)
  const composerHeight = useElementSize(composerRef).height
  // The frame's insets follow the tab it lies over: the previewed one, else the frontmost
  const showsFrame = peeked || (isActive && !framePeeked)
  useEffect(() => {
    if (!showsFrame) return
    if (hasComposer && composerHeight === 0) return // hidden tab measures 0, keep the last insets
    setFrameInsets(hasComposer ? { top: topInset, bottom: composerHeight, backdrop } : { top: 0, bottom: 0, backdrop })
  }, [showsFrame, hasComposer, composerHeight, topInset, backdrop, setFrameInsets])

  return (
    <div className="relative h-full w-full">
      <div ref={setSlot} className="absolute inset-0 flex items-center justify-center">
        {frameStatus === "unreachable" ? (
          <div className="flex flex-col items-center gap-2 text-xs text-ink-subtle">
            <p>omp-web is not reachable at {ompWebUrl}</p>
            <button type="button" onClick={reloadFrame} className="rounded px-2 py-1 text-ink-muted hover:bg-hover hover:text-ink">Retry</button>
          </div>
        ) : frameStatus === "loading" && (
          <p className="text-xs text-ink-subtle">Loading omp…</p>
        )}
      </div>
      {hasComposer && (
        <div ref={composerRef} className="pointer-events-none absolute inset-x-0 bottom-0 z-10">
          <div className={`absolute inset-x-0 bottom-0 h-48 bg-gradient-to-t ${backdrop === "surface" ? "from-surface via-surface/40" : "from-paper via-paper/40"} via-30% to-transparent`} />
          <div className="pointer-events-auto relative">
            <OmpComposer view={view} chatId={tab.chatId} slug={tab.projectSlug} shellCommands={shellCommands} insertion={insertion} onInserted={onInserted} lead={lead} onLeadSent={onLeadSent} />
          </div>
        </div>
      )}
    </div>
  )
}
