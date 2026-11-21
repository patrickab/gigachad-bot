import { useEffect, useState, type ReactNode } from "react"
import { anchorScene } from "deepfocus-ambient"
import { useElementSize } from "@/hooks/useElementSize"

/** Extra scroll padding below the last turn, on top of the floating composer's height. */
const BOTTOM_GAP_PX = 16
/** An empty conversation centers the composer this far above the column's bottom, as a share of its height. Slightly below the middle. */
const EMPTY_CENTER_FROM_BOTTOM = 0.45
const EMPTY_GREETING = "What's on your mind?"
/** Bottom padding inside the dock, under the composer box (`pb-6`). */
const DOCK_PADDING_BOTTOM_PX = 24
/** The backdrop's horizon lies this far below a lifted composer box, so the box floats just above the haze. */
const HORIZON_BELOW_COMPOSER_PX = 30

/** Where a conversation column's floating composer sits, from `useComposerDock`. */
export interface ComposerDockLayout {
  /** The column the composer floats in, `relative` and as tall as the conversation's area. */
  mountColumn: (element: HTMLDivElement | null) => void
  mountInputArea: (element: HTMLDivElement | null) => void
  /** Padding below the scrolling turns, so the last one clears the composer. */
  paddingBottom: number
  /** How far the composer is lifted from the bottom. */
  lift: number
  showGreeting: boolean
}

/**
 * The layout of a conversation column's floating composer, shared by chats and Pi sessions.
 *
 * - The turns scroll on under the composer, `paddingBottom` keeps the last one clear of it, also while the composer is hidden
 * - A conversation known to be `empty` lifts the composer to just below the column's middle, under the greeting. A loading one keeps it at the bottom
 * - A lifted composer anchors the ambient backdrop's scene: the horizon just below its box, the light behind the box's middle
 */
export function useComposerDock(empty: boolean): ComposerDockLayout {
  // Callback refs kept in state, not ref objects: the layout is read during render, which ref objects forbid
  const [column, mountColumn] = useState<HTMLDivElement | null>(null)
  const [inputArea, mountInputArea] = useState<HTMLDivElement | null>(null)
  // Preserve composer spacing while hidden so returning tabs do not jump.
  const inputAreaSize = useElementSize(inputArea)
  const [inputAreaHeight, setInputAreaHeight] = useState(0)
  if (inputAreaSize.height > 0 && inputAreaSize.height !== inputAreaHeight) setInputAreaHeight(inputAreaSize.height)
  const columnHeight = useElementSize(column).height
  const lift = empty && columnHeight > 0 ? Math.max(0, columnHeight * EMPTY_CENTER_FROM_BOTTOM - inputAreaHeight / 2) : 0
  const isLifted = lift > 0 && inputAreaHeight > 0
  useEffect(() => {
    if (!isLifted || !column) return
    return anchorScene(() => {
      const box = column.getBoundingClientRect()
      if (box.height === 0) return null
      const boxBottom = box.bottom - lift - DOCK_PADDING_BOTTOM_PX
      return {
        horizonY: boxBottom + HORIZON_BELOW_COMPOSER_PX,
        lightX: box.left + box.width / 2,
        lightY: boxBottom - (inputAreaHeight - DOCK_PADDING_BOTTOM_PX) / 2,
      }
    })
  }, [isLifted, column, lift, inputAreaHeight])
  // The greeting waits for the measured, centered composer, so it never flashes at the bottom
  return { mountColumn, mountInputArea, paddingBottom: inputAreaHeight + BOTTOM_GAP_PX, lift, showGreeting: isLifted }
}

/** The composer `children` floating at the bottom of `dock`'s column, with the greeting above it while lifted. `animate` eases its move to the bottom (a first send). */
export function ComposerDock({ dock, animate = false, children }: { dock: ComposerDockLayout; animate?: boolean; children: ReactNode }) {
  const { mountInputArea, lift, showGreeting } = dock
  return (
    <div
      ref={mountInputArea}
      className={`absolute bottom-0 left-0 right-0 z-10 pointer-events-none ${animate ? "transition-transform duration-200 ease-out motion-reduce:transition-none" : ""}`}
      style={{ transform: `translateY(${-lift}px)` }}
    >
      <p
        aria-hidden={!showGreeting}
        className={`absolute inset-x-0 bottom-full mb-5 text-center text-lg text-ink-subtle transition-opacity duration-200 ease-out motion-reduce:transition-none ${showGreeting ? "opacity-100" : "opacity-0"}`}
      >
        {EMPTY_GREETING}
      </p>
      <div className="relative px-4 pb-6 pointer-events-auto">
        {children}
      </div>
    </div>
  )
}
