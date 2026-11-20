import { useRef, useState, type ReactNode, type RefObject } from "react"
import { useElementSize } from "@/hooks/useElementSize"

/** Extra scroll padding below the last turn, on top of the floating composer's height. */
const BOTTOM_GAP_PX = 16
/** An empty conversation centers the composer this far above the column's bottom, as a share of its height. Slightly below the middle. */
const EMPTY_CENTER_FROM_BOTTOM = 0.45
const EMPTY_GREETING = "What's on your mind?"

/** Where a conversation column's floating composer sits, from `useComposerDock`. */
export interface ComposerDockLayout {
  /** The column the composer floats in, `relative` and as tall as the conversation's area. */
  columnRef: RefObject<HTMLDivElement | null>
  inputAreaRef: RefObject<HTMLDivElement | null>
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
 */
export function useComposerDock(empty: boolean): ComposerDockLayout {
  const columnRef = useRef<HTMLDivElement>(null)
  const inputAreaRef = useRef<HTMLDivElement>(null)
  // Preserve composer spacing while hidden so returning tabs do not jump.
  const inputAreaSize = useElementSize(inputAreaRef)
  const [inputAreaHeight, setInputAreaHeight] = useState(0)
  if (inputAreaSize.height > 0 && inputAreaSize.height !== inputAreaHeight) setInputAreaHeight(inputAreaSize.height)
  const columnHeight = useElementSize(columnRef).height
  const lift = empty && columnHeight > 0 ? Math.max(0, columnHeight * EMPTY_CENTER_FROM_BOTTOM - inputAreaHeight / 2) : 0
  // The greeting waits for the measured, centered composer, so it never flashes at the bottom
  return { columnRef, inputAreaRef, paddingBottom: inputAreaHeight + BOTTOM_GAP_PX, lift, showGreeting: lift > 0 && inputAreaHeight > 0 }
}

/** The composer `children` floating at the bottom of `dock`'s column, with the greeting above it while lifted. `animate` eases its move to the bottom (a first send). */
export function ComposerDock({ dock, animate = false, children }: { dock: ComposerDockLayout; animate?: boolean; children: ReactNode }) {
  return (
    <div
      ref={dock.inputAreaRef}
      className={`absolute bottom-0 left-0 right-0 z-10 pointer-events-none ${animate ? "transition-transform duration-200 ease-out motion-reduce:transition-none" : ""}`}
      style={{ transform: `translateY(${-dock.lift}px)` }}
    >
      <p
        aria-hidden={!dock.showGreeting}
        className={`absolute inset-x-0 bottom-full mb-5 text-center text-lg text-ink-subtle transition-opacity duration-200 ease-out motion-reduce:transition-none ${dock.showGreeting ? "opacity-100" : "opacity-0"}`}
      >
        {EMPTY_GREETING}
      </p>
      <div className="relative px-4 pb-6 pointer-events-auto">
        {children}
      </div>
    </div>
  )
}
