"use client"

import { useEffect, useEffectEvent, useLayoutEffect, useRef, useState } from "react"
import { createPortal } from "react-dom"
import { useTab } from "@/contexts/TabContext"
import { useEscapeLayer } from "@/hooks/useEscapeLayer"
import type { LearnView } from "@/lib/domain/types"
import { learnTitle } from "./items"
import { useLearn, useLearnDrive, useLearnFrameState } from "./LearnProvider"

/** `clip-path` covering the viewport except the box `left`/`top`/`right`/`bottom` (viewport pixels). */
function clipAround({ left, top, right, bottom }: DOMRect): string {
  return `polygon(evenodd, 0 0, 100% 0, 100% 100%, 0 100%, 0 0, ${left}px ${top}px, ${right}px ${top}px, ${right}px ${bottom}px, ${left}px ${bottom}px, ${left}px ${top}px)`
}

/**
 * The shell's half of a modal dialog open in the frame (`modal`), like omp's frame backdrop: the shell dims around the
 * frame, which dims itself under its dialog. A click on the dimmed shell or Escape in it closes the dialog (`onClose`),
 * Escape never reaches the shell's other handlers. The frame's box stays a hole, so the dialog keeps its clicks.
 */
function FrameModalScrim({ frameBox, onClose }: { frameBox: HTMLElement; onClose: () => void }) {
  const scrimRef = useRef<HTMLDivElement>(null)
  useEscapeLayer(onClose)
  useLayoutEffect(() => {
    const place = () => {
      if (scrimRef.current) scrimRef.current.style.clipPath = clipAround(frameBox.getBoundingClientRect())
    }
    place()
    const observer = new ResizeObserver(place)
    observer.observe(frameBox)
    window.addEventListener("resize", place)
    return () => {
      observer.disconnect()
      window.removeEventListener("resize", place)
    }
  }, [frameBox])
  return createPortal(<div ref={scrimRef} data-learn-modal-scrim="" onClick={onClose} className="fixed inset-0 z-[90] bg-backdrop" />, document.body)
}

/**
 * One learn tab: the slot the shared LearnHarness frame is laid over while the tab is frontmost.
 *
 * - A session the user opens or starts in a tab opened from a project joins that project, unless it already belongs to one
 * - LearnHarness's own composer and stats bar work inside the frame, the shell drives its model selector and chrome
 * - A modal dialog in the frame (settings, setup) dims the shell around the frame. A click on the dimmed shell or Escape
 *   closes it (`close-modal`), Escape never reaches the shell's handlers
 * - A frame at another protocol version keeps upstream's own UI and the tab asks for a LearnHarness restart
 */
export function LearnSurface({ view, onViewChange }: { view: LearnView; onViewChange: (view: LearnView) => void }) {
  const { tab, isActive, setTitle } = useTab()
  const { attachFrame, frameStatus, learnUrl, reloadFrame, sessions, sendToLearnFrame } = useLearn()
  const drive = useLearnDrive()
  const [slot, setSlot] = useState<HTMLDivElement | null>(null)
  const adoptView = useEffectEvent((next: LearnView) => {
    onViewChange(next)
    if (next.kind === "session" && tab.projectSlug) void sessions.claim(next.sessionId, tab.projectSlug)
  })

  useEffect(() => {
    if (!isActive || !slot || !learnUrl) return
    return attachFrame({ slot, view, onViewChange: (next) => adoptView(next) })
  }, [isActive, slot, view, learnUrl, attachFrame])

  const title = learnTitle(view, sessions.sessions)
  useEffect(() => {
    setTitle(title)
  }, [title, setTitle])

  const modal = useLearnFrameState("modal")
  const underModal = drive === "matched" && frameStatus === "ready" && isActive && !!modal?.open

  return (
    <div className="flex h-full w-full flex-col">
      {learnUrl && drive === "mismatch" && (
        <p role="status" className="shrink-0 border-b border-divider bg-surface px-4 py-1.5 text-center text-xs text-ink-muted">
          LearnHarness needs a restart: it runs another version than gigachad, so its own model menu and sidebar stay in use
        </p>
      )}
      <div className="relative min-h-0 flex-1">
        <div ref={setSlot} className="absolute inset-0 flex items-center justify-center text-xs text-ink-subtle">
          {!learnUrl ? <p>Learn is limited to host logins</p>
            : frameStatus === "unreachable" ? (
              <div className="flex flex-col items-center gap-2">
                <p>LearnHarness is not reachable at {learnUrl}</p>
                <button type="button" onClick={reloadFrame} className="rounded px-2 py-1 text-ink-muted hover:bg-hover hover:text-ink">Retry</button>
              </div>
            ) : frameStatus === "loading" && <p>Loading learn…</p>}
        </div>
        {underModal && slot && <FrameModalScrim frameBox={slot} onClose={() => sendToLearnFrame({ type: "close-modal" })} />}
      </div>
    </div>
  )
}
